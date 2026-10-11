from data import generate
from model import train, fairness_audit, save

df = generate()
model, metrics, test_df = train(df)
save(model)
print("Metrics:", {k: round(v, 3) for k, v in metrics.items()})
print("\nFairness audit:\n", fairness_audit(test_df).to_string(index=False))
