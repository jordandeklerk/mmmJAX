plan = mj.optimize_budget(
    model,
    results,
    quantity="mu",
    spend_constraint_lower=0.3,
    spend_constraint_upper=0.3,
    include_metrics=True,
)
mj.plot_budget_response(plan)
