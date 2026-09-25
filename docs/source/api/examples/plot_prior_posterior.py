prior_results = mj.sample_prior(model, priors, draws=500, seed=0)
mj.plot_prior_posterior(results, prior_results)
