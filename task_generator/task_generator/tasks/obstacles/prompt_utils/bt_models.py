from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel

# TreeNodesModel
# --------------
class InputPort(BaseModel):
    name: str
    type: str
    default: Optional[str] = None
    value: Optional[str] = None

    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="input_port", 
            attrib={
                "name": self.name, 
                "type": self.type
            }
        )
        if self.default:
            element.set("default", self.default)
        if self.value:
            element.set("value", self.value)

        return element

class OutputPort(BaseModel):
    name: str
    type: str

    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="output_port",
            attrib={
                "name": self.name,
                "type": self.type
            }
        )

        return element

class Condition(BaseModel):
    ID: str
    input_ports: Optional[List[InputPort]] = []
    output_port: Optional[List[OutputPort]] = []

    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="Condition",
            attrib={
                "ID": self.ID
            }
        )
        
        for ip in self.input_ports or []:
            ip: InputPort
            element.append(ip.to_xml())

        for op in self.output_port or []:
            op: OutputPort
            element.append(op.to_xml())
        
        return element


class Action(BaseModel):
    ID: str
    input_ports: Optional[List[InputPort]] = []
    output_port: Optional[List[OutputPort]] = []

    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="Action",
            attrib={
                "ID": self.ID
            }
        )

        for ip in self.input_ports or []:
            ip: InputPort
            element.append(ip.to_xml())

        for op in self.output_port or []:
            op: OutputPort
            element.append(op.to_xml())
        
        return element


class TreeNodesModel(BaseModel):
    conditions: Optional[List[Condition]] = []
    actions: Optional[List[Action]] = []

    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="TreeNodesModel",
            attrib={}
        )

        for action in self.actions:
            element.append(action.to_xml())

        for condition in self.conditions:
            element.append(condition.to_xml())

        return element

# BehaviorTree
# ------------
class TreeNode(BaseModel):
    ID: str
    name: str
    attributes: List[Dict]

    @abstractmethod
    def to_xml(self) -> ET.Element:
        raise NotImplementedError()

class DecorationNode(TreeNode):
    ID: Literal[
        "Fallback",
        "TimeDelayDecorator",
        "RetryUntilSuccessful"
    ]
    child_node: TreeNode

    def to_xml(self):
        element = ET.Element(
            tag=self.ID,
            attrib=self.attributes
        )

        element.append(self.child_node.to_xml())

        return element
    

class ControlNode(TreeNode):
    ID: Literal[
        "Sequence",
        "Fallback"
    ]
    children_nodes: List[TreeNode]

    def to_xml(self):
        element = ET.Element(
            tag=self.ID,
            attrib=self.attributes
        )

        for node in self.children_nodes:
            element.append(node.to_xml())

        return element


class LeafNode(ABC, TreeNode):
    def to_xml(self):
        element = ET.Element(
            tag=self.ID,
            attrib=self.attributes
        )

        return element
    

class ActionNode(LeafNode):
    ID: Literal[
        "FindNearestAgent",
        "SaySomething",
        "SetGroupId",
        "SetGoal",
        "StopMovement",
        "ResumeMovement",
        "StopAndWaitTimerAction",
        "ConversationFormation",
        "GoTo",
        "ApproachAgent",
        "ApproachRobot",
        "BlockRobot",
        "BlockAgent",
        "GroupWalk",
        "LookAtPoint",
        "LookAtAgent",
        "LookAtRobot",
        "FollowAgent",
    ]


class ConditionNode(LeafNode):
    ID: Literal[
        "RandomChanceCondition",
        "IsRobotFacingAgent",
        "IsAgentVisible",
        "IsRobotClose",
        "IsAgentClose",
        "IsAtPosition",
        "IsAnyoneSpeaking",
        "IsSpeaking",
        "IsAnyoneLookingAtMe",
        "IsLookingAtMe"
    ]


class BehaviorTree(BaseModel):
    ID: str
    children_nodes: List[TreeNode]

    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="BehaviorTree",
            attrib={
                "ID": self.ID
            }
        )

        for node in self.children_nodes:
            element.append(node.to_xml())

        return element

class Root(BaseModel):
    main_tree_to_execute: str
    BTCPP_format: str
    tree_nodes_model: TreeNodesModel
    behavior_trees: List[BehaviorTree]
    
    def to_xml(self) -> ET.Element:
        element = ET.Element(
            tag="root",
            attrib={
                "main_tree_to_execute": self.main_tree_to_execute,
                "BTCPP_format": self.BTCPP_format
            }
        )

        element.append(self.tree_nodes_model.to_xml())

        for behavior_tree in self.behavior_trees:
            element.append(behavior_tree.to_xml())

        return element