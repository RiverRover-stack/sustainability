"""
Source Package

Main package for the Smart AI Energy Predictor.
"""

# Re-export main modules for backward compatibility
from src.carbon import CarbonCalculator, calculate_carbon_footprint
from src.recommender import EnergyRecommender, get_recommendations_for_dashboard
from src.agent import EnergyAdvisorAgent, get_energy_agent
from src.data import generate_data, calculate_bill, prepare_features
from src.training import train_all_models, compare_models
