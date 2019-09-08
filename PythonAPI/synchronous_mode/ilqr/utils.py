def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    final_control = us[-1]
    if iteration_count % 30 == 0:
        print("iteration", iteration_count, info, J_opt, final_state, final_control)

def on_iteration_enhance(iteration_count, xs, us, J_opt, accepted, converged):
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    final_control = us[-1]
    if iteration_count % 30 == 0:
        print("iteration", iteration_count, info, J_opt, final_state, final_control)
    



def num_parameters(model):
    parameters = model.parameters()
    num_pars = 0
    for par in parameters:
        n_ = 1
        for i in list(par.size()):
            n_ *= i
        num_pars += n_
    return num_pars

def QP(model, P_qp, q_qp, G_qp, h_qp, A_gp = None, b_qp = None):
    P_qp = numpy_sparse_to_spmatrix(P_qp)
    q_qp = matrix(q_qp)
    G_qp = numpy_sparse_to_spmatrix(G_qp)
    h_qp = matrix(h_qp)
    print("Starting solving QP")
    solvers.options['feastol'] = 1e-5
    sol = solvers.qp(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp)
    
    theta_diffs = list(sol['x'])
    if theta_diffs is None:
        theta_diffs = np.zeros(num_parameters(model))
    return theta_diffs
