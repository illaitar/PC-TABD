"""SIN3D++ Conventions (fix before implementing).

0.1 Spatial:
  - u: pixel coords in reference (center) frame.
  - disp(u, t): where pixel u goes at time t (pixels, reference coords).
  - Backward warp (grid_sample): sample from source at u+disp into target u.
  - Forward splat: scatter source contributions into target grid.

0.2 Temporal:
  - t=0: center (reference) frame.
  - t in [-T/2, +T/2] for global shutter (we use normalized [-1, 1] -> map to [-T/2,+T/2]).
  - Shutter profile w(t) normalized so sum_i w_i approx 1.

0.3 Rolling shutter:
  - RS = time shift per row: t_eff(u, t) = t + dt(row(u)).
  - dt in seconds or fraction of exposure; keep units consistent.
  - rs_factor in [0,1], shutter_length relative -> dt = rs_factor * shutter_length * y_norm.
"""
