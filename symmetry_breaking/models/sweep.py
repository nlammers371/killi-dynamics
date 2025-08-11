from joblib import Parallel, delayed
from pde import ScalarField, FieldCollection, PDEBase, CartesianGrid

def make_1d_grid(length=3000, dx=10, periodic=True):
    N = int(length / dx)
    grid = CartesianGrid([[0, length]], shape=(N,), periodic=periodic)
    return grid


def run_simulation_1D(param_dict, grid, model_class, tracker_class, dt=10, T=36000, interval=100):

    model = model_class(**param_dict)
    state = model.get_state(grid)

    tracker = tracker_class(grid, interval=interval)
    _ = model.solve(state, t_range=T, dt=dt, tracker=tracker)

    # Merge input parameters and tracked metrics
    result = {
        **param_dict,
        **tracker.get_metrics(),  # You’ll define this method
    }

    return result
#
# results = Parallel(n_jobs=16)(
#     delayed(run_simulation)(param) for param in param_list
# )