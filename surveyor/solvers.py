"""Search solvers the drivers can plan with, selected by --solver.

cem, icem and mppi come from stable-worldmodel and share the CEM knobs the drivers
already expose (num_samples, n_steps, topk, var_scale, seed); the extra
iCEM and MPPI parameters stay at the library defaults (iCEM: noise_beta 2.0,
alpha 0.1, n_elite_keep 5, return_mean; MPPI: temperature 0.5) so a solver
swap changes exactly one thing. The router's c* probe (sources.py) keeps CEM unless --probe-solver match.

gd (battery N, 2026-09-15) is stable-worldmodel's GradientSolver at LeWM's own
config/eval/solver/adam.yaml: AdamW lr 0.1, 30 steps, 100 samples, no action
noise. It backpropagates the cost through the frozen predictor. The CEM knobs
are accepted and ignored, so gd rows run the official gradient planner, not a
CEM-shaped one.
"""

CHOICES = ("cem", "icem", "mppi", "gd")

GD_N_STEPS = 30
GD_NUM_SAMPLES = 100
GD_LR = 0.1


def add_solver_arg(p):
    p.add_argument("--solver", choices=CHOICES, default="cem",
                   help="planning solver over the world model: cem (default), icem, mppi, "
                        "gd (LeWM adam.yaml gradient planner)")


def _gd_class():
    import torch
    import stable_worldmodel as swm

    class OfficialGDSolver(swm.solver.GradientSolver):
        """GradientSolver at LeWM config/eval/solver/adam.yaml; CEM-only knobs ignored."""

        def __init__(self, model, batch_size=1, device="cuda", seed=1234, **_cem_knobs):
            super().__init__(model=model, n_steps=GD_N_STEPS, batch_size=batch_size,
                             num_samples=GD_NUM_SAMPLES, action_noise=0.0, device=device,
                             seed=seed, optimizer_cls=torch.optim.AdamW,
                             optimizer_kwargs={"lr": GD_LR})

        def init_action(self, n_envs, actions=None):
            # patch N2 (16 Sep 2026): upstream only .to(device) when padding the horizon.
            if actions is not None:
                actions = actions.to(self.device, dtype=self.dtype)
            return super().init_action(n_envs, actions)

        def solve(self, *args, **kwargs):
            # patch N3 (16 Sep 2026): callers such as the c* probe run under @torch.no_grad();
            # the gradient solver needs autograd, so re-enable it for the solve only.
            with torch.enable_grad():
                return super().solve(*args, **kwargs)

    return OfficialGDSolver


def make_solver(name, cost_model, *, num_samples, var_scale, n_steps, topk, device, seed,
                batch_size=1):
    import stable_worldmodel as swm
    kw = dict(model=cost_model, batch_size=batch_size, num_samples=num_samples,
              var_scale=var_scale, n_steps=n_steps, topk=topk, device=device, seed=seed)
    if name == "gd":
        print(f"[solver] gd (OfficialGDSolver, AdamW lr={GD_LR}) num_samples={GD_NUM_SAMPLES} "
              f"n_steps={GD_N_STEPS} seed={seed}; ignored CEM knobs num_samples={num_samples} "
              f"n_steps={n_steps} topk={topk} var_scale={var_scale}")
        return _gd_class()(**kw)
    cls = {"cem": swm.solver.CEMSolver, "icem": swm.solver.ICEMSolver,
           "mppi": swm.solver.MPPISolver}[name]
    print(f"[solver] {name} ({cls.__name__}) num_samples={num_samples} n_steps={n_steps} "
          f"topk={topk} var_scale={var_scale} seed={seed}")
    return cls(**kw)


def solver_class(name):
    """The solver class by name, without the banner print: the router's c* probe
    builds one per replan (patch_L, 2026-09-14)."""
    import stable_worldmodel as swm
    if name == "gd":
        return _gd_class()
    return {"cem": swm.solver.CEMSolver, "icem": swm.solver.ICEMSolver,
            "mppi": swm.solver.MPPISolver}[name]
