"""
Pokemon Red Reward Library

Comprehensive collection of reward components organized by category.
"""

from .pallet_town_rewards import *
from .exploration_rewards import *
from .social_rewards import *
from .pokemon_rewards import *
from .battle_rewards import *
from .story_rewards import *
from .economy_rewards import *
from .efficiency_rewards import *
from .novelty_rewards import *
from .adaptive_rewards import *
from .composite_rewards import *

__all__ = [
    # Pallet Town Early Game
    "LeaveStartingRoomReward",
    "TalkToMomReward",
    "InteractWithTVReward",
    "CheckComputerReward",
    "HouseFullyExploredReward",
    "ExitHouseReward",
    "ExploreTownReward",
    "TalkToNPCsReward",
    "OakLabDiscoveryReward",
    "AttemptRoute1Reward",
    "OakEncounterReward",
    "FollowOakToLabReward",
    "ChooseStarterPokemonReward",
    "RivalEncounterReward",
    "FirstPokemonBattleReward",
    "MenuDiscoveryReward",
    "PokemonMenuReward",
    "BagDiscoveryReward",
    "SaveGameReward",
    "TryAllDirectionsReward",
    "DoorInteractionReward",
    "ObjectInteractionReward",
    "SignReadingReward",
    "CompleteTownExplorationReward",
    "AllNPCsTalkedToReward",
    "ReadyForAdventureReward",
    # Exploration
    "NewAreaDiscoveryReward",
    "AreaCompletionReward",
    "RouteCompletionReward",
    "BuildingEntryReward",
    "HiddenAreaDiscoveryReward",
    "HiddenItemFoundReward",
    "FirstItemOfTypeReward",
    "RareItemDiscoveryReward",
    "KeyItemAcquisitionReward",
    "FirstWarpUsageReward",
    "PCUsageReward",
    "VendingMachineReward",
    # Social & NPC
    "NewNPCConversationReward",
    "HelpfulInformationReceivedReward",
    "StoryDialogueProgressionReward",
    "ProfessorOakInteractionsReward",
    "NPCGiftReceivedReward",
    "TradeCompletionReward",
    "NameRaterUsageReward",
    # Pokemon Collection
    "FirstPokemonCaughtReward",
    "NewSpeciesCaughtReward",
    "RarePokemonCaughtReward",
    "EvolutionStonePokemonReward",
    "PokedexMilestonesReward",
    "AreaPokedexCompletionReward",
    "TypeCollectionReward",
    "PokemonEvolutionReward",
    "LevelMilestonesReward",
    "MoveLearningReward",
    "TMHMTeachingReward",
    # Battle & Combat
    "WildPokemonDefeatedReward",
    "TrainerBattleVictoryReward",
    "GymLeaderVictoryReward",
    "EliteFourMemberVictoryReward",
    "ChampionVictoryReward",
    "TypeAdvantageUsageReward",
    "CriticalHitReward",
    "StatusEffectUsageReward",
    "OHKOReward",
    "FlawlessVictoryReward",
    "UnderleveledVictoryReward",
    "BattleStreakReward",
    # Story & Achievement
    "GymBadgeEarnedReward",
    "HMAcquisitionReward",
    "EliteFourAccessReward",
    "HallOfFameEntryReward",
    "RivalBattleCompletionReward",
    "TeamRocketDefeatReward",
    "LegendaryEncounterReward",
    "SilphCoCompletionReward",
    "SafariZoneSuccessReward",
    "GameCornerPrizesReward",
    "FossilRevivalReward",
    # Economy & Resources
    "FirstEarningsReward",
    "WealthMilestonesReward",
    "SmartPurchasesReward",
    "RarePurchaseReward",
    "InventoryOrganizationReward",
    "HealingItemUsageReward",
    "PokeballEfficiencyReward",
    # Efficiency & Optimization
    "FastTravelUsageReward",
    "OptimalRoutingReward",
    "PuzzleSolvingReward",
    "MoveEffectivenessReward",
    "EvolutionTimingReward",
    "HMUsageReward",
    # Novelty & Exploration
    "FirstBattleReward",
    "FirstPokemonCenterVisitReward",
    "FirstPokemartPurchaseReward",
    "FirstSaveReward",
    "MenuExplorationReward",
    "ButtonDiscoveryReward",
    "FeatureDiscoveryReward",
    # Adaptive & Learning
    "MistakeRecoveryReward",
    "StrategyAdaptationReward",
    "ResourceConservationReward",
    "PatternRecognitionReward",
    "RouteOptimizationReward",
    "BattlePreparationReward",
    # Composite & Milestone
    "PerfectGymRunReward",
    "AreaMasteryReward",
    "SpeedrunMilestonesReward",
    "ExplorationStreakReward",
    "BattleWinStreakReward",
    "PerfectDayReward",
]
