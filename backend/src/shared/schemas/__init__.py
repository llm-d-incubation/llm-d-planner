"""Shared Pydantic schemas for NeuralNav backend.

This module provides all data schemas used across the application,
organized by domain:
- intent: User intent and conversation schemas
- specification: Traffic profile and SLO target schemas
- recommendation: GPU config, scores, and recommendation schemas
"""

from .intent import ConversationMessage, DeploymentIntent
from .recommendation import (
    ConfigurationScores,
    DeploymentRecommendation,
    GPUConfig,
    RankedRecommendationsResponse,
)
from .specification import DeploymentSpecification, SLOTargets, TrafficProfile
