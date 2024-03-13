import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.models.decoders.action_decoder import ActionDecoder
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.distributions as D

logger = logging.getLogger(__name__)


class SimpleIL(pl.LightningModule, CalvinBaseModel):
    def __init__(
        self,
        perceptual_encoder: DictConfig,
        visual_goal: DictConfig,
        language_goal: DictConfig,
        action_decoder: DictConfig,
        optimizer: DictConfig,
        replan_freq: int = 30,
    ):
        super(SimpleIL, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)
        self.setup_input_sizes(
            self.perceptual_encoder,
            visual_goal,
            action_decoder,
        )

        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None

        # policy network
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)

        self.modality_scope = "lang"
        self.optimizer_config = optimizer
        # workaround to resolve hydra config file before calling save_hyperparams  until they fix this issue upstream
        # without this, there is conflict between lightning and hydra
        action_decoder.out_features = action_decoder.out_features

        self.optimizer_config["lr"] = self.optimizer_config["lr"]
        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.lang_embeddings = None

    @staticmethod
    def setup_input_sizes(
        perceptual_encoder,
        visual_goal,
        action_decoder,
    ):
        visual_goal.in_features = perceptual_encoder.latent_size
        action_decoder.perceptual_features = perceptual_encoder.latent_size

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'decomp_obs' (dict):
                        - 'seg_static' (Tensor): Segmentation mask of static camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.


        Returns:
            loss tensor
        """
        total_loss = torch.tensor(0.0).to(self.device)

        for self.modality_scope, dataset_batch in batch.items():
            # check if there is any NaN values in the batch
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["decomp_obs"], dataset_batch["robot_obs"]
            )
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
            else:
                latent_goal = self.visual_goal(perceptual_emb[:, -1])
                
            mod_loss = self.action_decoder.loss(None, perceptual_emb, latent_goal, dataset_batch["actions"])

            total_loss += mod_loss
            self.log(f"train/total_loss_{self.modality_scope}", mod_loss, on_step=False, on_epoch=True)
            
        total_loss = total_loss / len(batch)  # divide accumulated gradients by number of datasets
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)
        return total_loss


    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.

        Args:
            batch (dict):
                - 'vis' (dict):
                    - 'rgb_obs' (dict):
                        - 'rgb_static' (Tensor): RGB camera image of static camera
                        - ...
                    - 'depth_obs' (dict):
                        - 'depth_static' (Tensor): Depth camera image of depth camera
                        - ...
                    - 'decomp_obs' (dict):
                        - 'seg_static' (Tensor): Segmentation mask of static camera
                        - ...
                    - 'robot_obs' (Tensor): Proprioceptive state observation.
                    - 'actions' (Tensor): Ground truth actions.
                    - 'state_info' (dict):
                        - 'robot_obs' (Tensor): Unnormalized robot states.
                        - 'scene_obs' (Tensor): Unnormalized scene states.
                    - 'idx' (LongTensor): Episode indices.
                - 'lang' (dict):
                    Like 'vis' but with additional keys:
                        - 'language' (Tensor): Embedded Language labels.
                        - 'use_for_aux_lang_loss' (BoolTensor): Mask of which sequences in the batch to consider for
                            auxiliary loss.
            batch_idx (int): Integer displaying index of this batch.

        Returns:
            Dictionary containing losses and the sampled plans of plan recognition and plan proposal networks.
        """
        output = {}
        for self.modality_scope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(
                dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["decomp_obs"], dataset_batch["robot_obs"]
            )
            latent_goal = (
                self.visual_goal(perceptual_emb[:, -1])
                if "vis" in self.modality_scope
                else self.language_goal(dataset_batch["lang"])
            )
            
            action_loss, sample_act = self.action_decoder.loss_and_act(
                None, perceptual_emb, latent_goal, dataset_batch["actions"]
            )

            mae = torch.nn.functional.l1_loss(
                sample_act[..., :-1], dataset_batch["actions"][..., :-1], reduction="none"
            )  # (batch, seq, 6)
            mae = torch.mean(mae, 1)  # (batch, 6)
            # gripper action
            gripper_discrete = sample_act[..., -1]
            gt_gripper_act = dataset_batch["actions"][..., -1]
            m = gripper_discrete > 0
            gripper_discrete[m] = 1
            gripper_discrete[~m] = -1
            gripper_sr = torch.mean((gt_gripper_act == gripper_discrete).float())        
            
            output[f"val_action_loss_{self.modality_scope}"] = action_loss
            output[f"mae_{self.modality_scope}"] = mae
            output[f"gripper_sr{self.modality_scope}"] = gripper_sr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]

        return output

    def validation_epoch_end(self, validation_step_outputs):
        val_total_act_loss = torch.tensor(0.0).to(self.device)
        val_total_mae = torch.tensor(0.0).to(self.device)
        val_pos_mae = torch.tensor(0.0).to(self.device)
        val_orn_mae = torch.tensor(0.0).to(self.device)
        val_grip_sr = torch.tensor(0.0).to(self.device)
        for mod in self.trainer.datamodule.modalities:
            act_loss = torch.stack([x[f"val_action_loss_{mod}"] for x in validation_step_outputs]).mean()
            mae = torch.cat([x[f"mae_{mod}"] for x in validation_step_outputs])
            pp_mae_mean = mae.mean()
            pos_mae = mae[..., :3].mean()
            orn_mae = mae[..., 3:6].mean()
            grip_sr = torch.stack([x[f"gripper_sr{mod}"] for x in validation_step_outputs]).mean()
            val_total_mae += pp_mae_mean
            val_pos_mae += pos_mae
            val_orn_mae += orn_mae
            val_grip_sr += grip_sr
            val_total_act_loss += act_loss

            self.log(f"val_act/{mod}_act_loss", act_loss, sync_dist=True)
            self.log(f"val_total_mae/{mod}_total_mae", pp_mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{mod}_pos_mae", pos_mae, sync_dist=True)
            self.log(f"val_orn_mae/{mod}_orn_mae", orn_mae, sync_dist=True)
            self.log(f"val_grip/{mod}_grip_sr", grip_sr, sync_dist=True)
        self.log(
            "val_act/action_loss", val_total_act_loss / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log(
            "val_total_mae/total_mae", val_total_mae / len(self.trainer.datamodule.modalities), sync_dist=True
        )
        self.log("val_pos_mae/pos_mae", val_pos_mae / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_orn_mae/orn_mae", val_orn_mae / len(self.trainer.datamodule.modalities), sync_dist=True)
        self.log("val_grip/grip_sr", val_grip_sr / len(self.trainer.datamodule.modalities), sync_dist=True)

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """
        Do one step of inference with the model.

        Args:
            obs (dict): Observation from environment.
            goal (str or dict): The goal as a natural language instruction or dictionary with goal images.

        Returns:
            Predicted action.
        """
        # replan every replan_freq steps (default 30 i.e every second)
        if self.rollout_step_counter % self.replan_freq == 0:
            if isinstance(goal, str):
                embedded_lang = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()
                self.latent_goal = self.get_goal_lang(obs, embedded_lang)
            else:
                self.latent_goal = self.get_goal_vision(obs, goal)
        # use plan to predict actions with current observations
        action = self.predict(obs, self.latent_goal, None)
        self.rollout_step_counter += 1
        return action

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}

    def predict(
        self,
        obs: Dict[str, Any],
        latent_goal: torch.Tensor,
        sampled_plan: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass observation, goal and plan through decoder to get predicted action.

        Args:
            obs: Observation from environment.
            latent_goal: Encoded goal.
            sampled_plan: Sampled plan proposal plan.

        Returns:
            Predicted action.
        """
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(obs["rgb_obs"], obs["depth_obs"], obs["decomp_obs"], obs["robot_obs"])
            action = self.action_decoder.act(sampled_plan, perceptual_emb, latent_goal)

        return action

    def get_goal_vision(self, obs: dict, goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(obs["rgb_obs"]) == len(goal["rgb_obs"])
        assert len(obs["depth_obs"]) == len(goal["depth_obs"])
        assert len(obs["decomp_obs"]) == len(goal["decomp_obs"])
        imgs = {k: torch.cat([v, goal["rgb_obs"][k]], dim=1) for k, v in obs["rgb_obs"].items()}  # (1, 2, C, H, W)
        depth_imgs = {k: torch.cat([v, goal["depth_obs"][k]], dim=1) for k, v in obs["depth_obs"].items()}
        decomp_imgs = {k: torch.cat([v, goal["decomp_obs"][k]], dim=1) for k, v in obs["decomp_obs"].items()}
        state = torch.cat([obs["robot_obs"], goal["robot_obs"]], dim=1)
        with torch.no_grad():
            perceptual_emb = self.perceptual_encoder(imgs, depth_imgs, decomp_imgs, state)
            latent_goal = self.visual_goal(perceptual_emb[:, -1])
        self.action_decoder.clear_hidden_state()
        return latent_goal

    def get_goal_lang(self, obs: dict, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            latent_goal = self.language_goal(goal)
        self.action_decoder.clear_hidden_state()
        return latent_goal

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        logger.info(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")
