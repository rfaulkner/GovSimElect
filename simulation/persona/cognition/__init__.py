"""Cognition components for persona agents."""

from simulation.persona.cognition import act
from simulation.persona.cognition import converse
from simulation.persona.cognition import perceive
from simulation.persona.cognition import plan
from simulation.persona.cognition import reflect
from simulation.persona.cognition import retrieve
from simulation.persona.cognition import store
from simulation.persona import common

ActComponent = act.ActComponent
ConverseComponent = converse.ConverseComponent
PerceiveComponent = perceive.PerceiveComponent
PersonaOberservation = common.PersonaOberservation
PlanComponent = plan.PlanComponent
ReflectComponent = reflect.ReflectComponent
RetrieveComponent = retrieve.RetrieveComponent
StoreComponent = store.StoreComponent
