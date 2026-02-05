"""Trajectory encoding/decoding for storage."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
import json


class TrajectoryCodec:
    """Encode/decode trajectories and metadata to/from .npy files.
    
    Format: Single .npy file containing a dict with:
        - cam_traj: [N, H, W, 2] camera trajectory
        - obj_traj: [N, H, W, 2] object trajectory (optional)
        - traj: [N, H, W, 2] combined trajectory
        - visibility: [N, H, W] visibility map (optional)
        - params: dict of generation parameters
        - meta: additional metadata
    """
    
    VERSION = 1
    
    @classmethod
    def encode(
        cls,
        cam_traj: np.ndarray,
        obj_traj: Optional[np.ndarray] = None,
        traj: Optional[np.ndarray] = None,
        visibility: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> Dict[str, Any]:
        """Encode trajectories to a dict for saving.
        
        Args:
            cam_traj: [N, H, W, 2] camera trajectory
            obj_traj: [N, H, W, 2] object trajectory
            traj: [N, H, W, 2] combined trajectory
            visibility: [N, H, W] visibility map
            params: generation parameters
            meta: additional metadata
            compress: use float16 for compression
        
        Returns:
            dict ready for np.save
        """
        dtype = np.float16 if compress else np.float32
        
        data = {
            "version": cls.VERSION,
            "cam_traj": cam_traj.astype(dtype),
        }
        
        if obj_traj is not None:
            data["obj_traj"] = obj_traj.astype(dtype)
        
        if traj is not None:
            data["traj"] = traj.astype(dtype)
        elif obj_traj is not None:
            data["traj"] = (cam_traj + obj_traj).astype(dtype)
        else:
            data["traj"] = cam_traj.astype(dtype)
        
        if visibility is not None:
            data["visibility"] = visibility.astype(dtype)
        
        if params is not None:
            data["params"] = cls._serialize_params(params)
        
        if meta is not None:
            data["meta"] = meta
        
        return data
    
    @classmethod
    def decode(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decode trajectories from loaded dict.
        
        Args:
            data: dict loaded from .npy
        
        Returns:
            dict with numpy arrays and params
        """
        result = {
            "cam_traj": data["cam_traj"].astype(np.float32),
            "traj": data["traj"].astype(np.float32),
        }
        
        if "obj_traj" in data:
            result["obj_traj"] = data["obj_traj"].astype(np.float32)
        
        if "visibility" in data:
            result["visibility"] = data["visibility"].astype(np.float32)
        
        if "params" in data:
            result["params"] = cls._deserialize_params(data["params"])
        
        if "meta" in data:
            result["meta"] = data["meta"]
        
        result["version"] = data.get("version", 0)
        
        return result
    
    @classmethod
    def save(cls, path: Union[str, Path], **kwargs) -> None:
        """Save trajectories to .npy file."""
        data = cls.encode(**kwargs)
        np.save(path, data, allow_pickle=True)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Load trajectories from .npy file."""
        data = np.load(path, allow_pickle=True).item()
        return cls.decode(data)
    
    @staticmethod
    def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize params to JSON-safe types."""
        serialized = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                serialized[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serialized[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif hasattr(v, "item"):
                serialized[k] = v.item()
            else:
                serialized[k] = v
        return serialized
    
    @staticmethod
    def _deserialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize params back to numpy/python types."""
        return params  # Already in python types
