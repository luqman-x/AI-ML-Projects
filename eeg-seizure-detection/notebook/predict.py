import pickle
import numpy as np

model = pickle.load(open("notebook/seizureWatch.ipynb.pkl", "rb"))

sample = np.random.rand(1,178)

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction:", prediction)
print("Probability:", probability)