from imageai.Detection.Custom import DetectionModelTrainer

trainer=DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer = DetectionModelTrainer()
trainer.setDataDirectory(data_directory="headsets2")
trainer.setTrainConfig(object_names_array=["surfer"], batch_size=4, num_expexriments=200, 
train_from_pretrained_model="/Users/hollands/dev/cfmodelserver/models/yolo-tiny.h5")
trainer.trainModel()