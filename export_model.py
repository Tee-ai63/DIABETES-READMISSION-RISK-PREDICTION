import joblib, json
pipeline = joblib.load("readmission_model.pkl")
preprocessor = pipeline.named_steps["preprocessor"]
num_features = list(preprocessor.transformers_[0][2])
cat_encoder  = preprocessor.transformers_[1][1].named_steps["onehot"]
cat_features = list(preprocessor.transformers_[1][2])
cat_feature_names = list(cat_encoder.get_feature_names_out(cat_features))
json.dump(num_features + cat_feature_names, open("feature_names.json", "w"))
print("done")