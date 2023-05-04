# DAAN572_RL_Proj
Final project for DAAN572-Reinforcement Learning - Model of Simglucose environment

#Environment 
A simulated one-day time period for 30 virtual diabetic adult and adolescent individuals. Each individual will eat meals at random and will have varied responses to the introduction of glucose and insulin. If a patient’s blood glucose goes into a failure state, they would receive instruction to eat food or manually inject insulin themselves. This environment may be characterized as a finite Markov Decision Process (MDP).

#States
Blood Glucose (measured as milligrams per deciliter mg/dl), represented as a continuous value with a clinically detectable range of 0 to 600. Healthy glucose levels are between 60 and 120. If the most recent glucose reading is below 50 or above 400 this may be considered a failure state. 

#Actions
The agent will have the ability to introduce between 0 and 30 units of basal insulin to the blood stream through an implanted device with the level of injected insulin determined about every three minutes when the CGM returns a reading.

#Rewards
The relative change in value of the Blood Glucose Risk Index (BGRI) as defined by Statistical Tools to Analyze Continuous Glucose Monitor Data (Clarke 2009). Computed as follows:
