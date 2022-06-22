using TensorBoardLogger, ValueHistories

# tb_logger = TBReader("C:/Users/hunte/Documents/Research/Preference_Elicitation/preference_elicitation/network_training/events.out.tfevents.1.655284723703e9.BlackTech")
path = "C:/Users/hunte/Documents/Research/Preference_Elicitation/preference_elicitation/network_training"
hist = convert(MVHistory, tb_logger)