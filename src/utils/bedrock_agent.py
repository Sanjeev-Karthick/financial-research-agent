"""
Bedrock Agent Helper Module

This module provides utility classes for creating and managing Amazon Bedrock Agents.
It simplifies the process of setting up multi-agent collaboration patterns including
supervisor agents that orchestrate specialized sub-agents.

Classes:
    Agent: Base class for creating individual Bedrock agents
    SupervisorAgent: Orchestrates multiple sub-agents in a supervisor pattern
    Task: Represents a task that can be executed by an agent
    Guardrail: Defines content filtering rules for agents
    AgentsHelper: Low-level helper for agent CRUD operations
"""

import boto3
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from botocore.exceptions import ClientError

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients - these will use default credential chain
sts_client = boto3.client("sts")
bedrock_agent_client = boto3.client("bedrock-agent")
bedrock_runtime_client = boto3.client("bedrock-agent-runtime")
iam_client = boto3.client("iam")

# Get AWS account info for resource ARNs
caller_identity = sts_client.get_caller_identity()
account_id = caller_identity["Account"]
region = boto3.session.Session().region_name or "us-east-1"

logger.info(f"Initialized Bedrock Agent helper for account {account_id} in region {region}")


@dataclass
class Guardrail:
    """
    Represents a content filtering guardrail for Bedrock agents.
    
    Guardrails can block certain topics or content from being processed by agents,
    helping ensure responsible AI usage in financial applications.
    
    Args:
        name: Unique identifier for this guardrail
        topic_name: Name of the topic being filtered
        description: Human-readable description of the guardrail purpose
        denied_topics: List of topics/keywords to block
        blocked_input_response: Message shown when input is blocked
        verbose: Whether to log detailed guardrail operations
    """
    name: str
    topic_name: str
    description: str
    denied_topics: List[str] = field(default_factory=list)
    blocked_input_response: str = "This topic is not allowed."
    verbose: bool = False
    
    # These get populated after creation
    guardrail_id: Optional[str] = None
    guardrail_arn: Optional[str] = None
    guardrail_version: str = "DRAFT"
    
    def __post_init__(self):
        """Create the guardrail in Bedrock after initialization."""
        self._create_guardrail()
    
    def _create_guardrail(self):
        """
        Creates the guardrail in Amazon Bedrock.
        
        This method handles the API call to create a new guardrail with
        the specified topic filtering configuration.
        """
        bedrock_client = boto3.client("bedrock")
        
        try:
            # Build the topic policy configuration
            topic_policy = {
                "topicsConfig": [
                    {
                        "name": self.topic_name,
                        "definition": self.description,
                        "examples": self.denied_topics,
                        "type": "DENY"
                    }
                ]
            }
            
            response = bedrock_client.create_guardrail(
                name=self.name,
                description=self.description,
                topicPolicyConfig=topic_policy,
                blockedInputMessaging=self.blocked_input_response,
                blockedOutputsMessaging=self.blocked_input_response,
            )
            
            self.guardrail_id = response["guardrailId"]
            self.guardrail_arn = response["guardrailArn"]
            
            if self.verbose:
                logger.info(f"Created guardrail '{self.name}' with ID: {self.guardrail_id}")
                
        except ClientError as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Guardrail '{self.name}' already exists, retrieving existing...")
                self._get_existing_guardrail()
            else:
                raise
    
    def _get_existing_guardrail(self):
        """Retrieve an existing guardrail by name."""
        bedrock_client = boto3.client("bedrock")
        
        response = bedrock_client.list_guardrails()
        for guardrail in response.get("guardrails", []):
            if guardrail["name"] == self.name:
                self.guardrail_id = guardrail["id"]
                self.guardrail_arn = guardrail["arn"]
                break


@dataclass 
class Task:
    """
    Represents a task that can be assigned to an agent.
    
    Tasks encapsulate the work that needs to be done and can be tracked
    for progress and completion status.
    
    Args:
        description: What the task should accomplish
        expected_output: What the result should look like
        agent: Optional agent assignment
    """
    description: str
    expected_output: str
    agent: Optional["Agent"] = None
    status: str = "pending"
    result: Optional[str] = None
    
    def complete(self, result: str):
        """Mark the task as complete with the given result."""
        self.status = "completed"
        self.result = result
        
    def fail(self, error: str):
        """Mark the task as failed with the given error message."""
        self.status = "failed"
        self.result = error


class AgentsHelper:
    """
    Low-level helper class for Bedrock agent CRUD operations.
    
    This class provides methods for creating, updating, and deleting
    Bedrock agents and their associated IAM roles.
    """
    
    def __init__(self):
        self.bedrock_agent = boto3.client("bedrock-agent")
        self.iam = boto3.client("iam")
        
    def create_agent_role(self, agent_name: str) -> str:
        """
        Creates an IAM role for a Bedrock agent.
        
        The role includes trust policy for Bedrock service and basic
        permissions needed for agent execution.
        
        Args:
            agent_name: Name of the agent (used in role name)
            
        Returns:
            ARN of the created role
        """
        role_name = f"AmazonBedrockExecutionRoleForAgents_{agent_name}"
        
        # Trust policy allowing Bedrock to assume this role
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": account_id},
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:agent/*"
                        }
                    }
                }
            ]
        }
        
        try:
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"Execution role for Bedrock agent {agent_name}"
            )
            role_arn = response["Role"]["Arn"]
            
            # Attach basic Bedrock permissions
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
            )
            
            # Give IAM time to propagate
            logger.info(f"Created IAM role {role_name}, waiting for propagation...")
            time.sleep(10)
            
            return role_arn
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            logger.info(f"Role {role_name} already exists, using existing role")
            return f"arn:aws:iam::{account_id}:role/{role_name}"
    
    def delete_agent(
        self, 
        agent_name: str, 
        delete_role_flag: bool = True, 
        verbose: bool = False
    ):
        """
        Deletes a Bedrock agent and optionally its IAM role.
        
        Args:
            agent_name: Name of the agent to delete
            delete_role_flag: Whether to also delete the IAM role
            verbose: Whether to log detailed operations
        """
        try:
            # First, find the agent by name
            response = self.bedrock_agent.list_agents()
            agent_id = None
            
            for agent in response.get("agentSummaries", []):
                if agent["agentName"] == agent_name:
                    agent_id = agent["agentId"]
                    break
            
            if agent_id:
                if verbose:
                    logger.info(f"Deleting agent {agent_name} (ID: {agent_id})")
                    
                self.bedrock_agent.delete_agent(agentId=agent_id)
                
                if verbose:
                    logger.info(f"Successfully deleted agent {agent_name}")
            else:
                if verbose:
                    logger.info(f"Agent {agent_name} not found, skipping deletion")
                    
            # Delete the IAM role if requested
            if delete_role_flag:
                role_name = f"AmazonBedrockExecutionRoleForAgents_{agent_name}"
                try:
                    # Detach policies before deleting role
                    policies = self.iam.list_attached_role_policies(RoleName=role_name)
                    for policy in policies.get("AttachedPolicies", []):
                        self.iam.detach_role_policy(
                            RoleName=role_name, 
                            PolicyArn=policy["PolicyArn"]
                        )
                    
                    self.iam.delete_role(RoleName=role_name)
                    
                    if verbose:
                        logger.info(f"Deleted IAM role {role_name}")
                        
                except self.iam.exceptions.NoSuchEntityException:
                    if verbose:
                        logger.info(f"Role {role_name} not found, skipping")
                        
        except ClientError as e:
            logger.error(f"Error deleting agent {agent_name}: {e}")
            raise


# Global helper instance for convenience
agents_helper = AgentsHelper()


class Agent:
    """
    Represents an Amazon Bedrock Agent.
    
    This class provides a high-level interface for creating and managing
    Bedrock agents, including their tools, instructions, and knowledge bases.
    
    Example:
        agent = Agent.create(
            name="my_agent",
            role="Financial Analyst",
            goal="Analyze stock performance",
            instructions="You are a helpful financial analyst...",
            llm="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
    """
    
    # Class-level flag to force recreation of existing agents
    _force_recreate = False
    
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        instructions: str,
        llm: str,
        tools: Optional[List[Dict]] = None,
        kb_id: Optional[str] = None,
        guardrail: Optional[Guardrail] = None,
        verbose: bool = True
    ):
        """
        Initialize an Agent instance.
        
        Args:
            name: Unique name for the agent
            role: Description of the agent's role
            goal: What the agent is trying to accomplish
            instructions: Detailed instructions for the agent
            llm: Model ID to use for the agent
            tools: Optional list of tool configurations
            kb_id: Optional knowledge base ID to attach
            guardrail: Optional guardrail for content filtering
            verbose: Whether to log detailed operations
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.instructions = instructions
        self.llm = llm
        self.tools = tools or []
        self.kb_id = kb_id
        self.guardrail = guardrail
        self.verbose = verbose
        
        # These get populated after creation
        self.agent_id: Optional[str] = None
        self.agent_arn: Optional[str] = None
        self.agent_alias_id: Optional[str] = None
        self.agent_alias_arn: Optional[str] = None
        
    @classmethod
    def set_force_recreate_default(cls, value: bool):
        """Set whether to force recreation of existing agents."""
        cls._force_recreate = value
        
    @classmethod
    def create(
        cls,
        name: str,
        role: str,
        goal: str,
        instructions: str,
        llm: str,
        tools: Optional[List[Dict]] = None,
        kb_id: Optional[str] = None,
        guardrail: Optional[Guardrail] = None,
        verbose: bool = True
    ) -> "Agent":
        """
        Factory method to create and deploy a new agent.
        
        This method handles all the steps needed to create a fully
        functional Bedrock agent including IAM role creation, agent
        creation, tool attachment, and alias creation.
        
        Returns:
            Fully initialized Agent instance ready for invocation
        """
        agent = cls(
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            llm=llm,
            tools=tools,
            kb_id=kb_id,
            guardrail=guardrail,
            verbose=verbose
        )
        
        agent._deploy()
        return agent
    
    def _deploy(self):
        """Deploy the agent to Amazon Bedrock."""
        if self.verbose:
            logger.info(f"Creating agent '{self.name}'...")
            
        # Create IAM role for the agent
        role_arn = agents_helper.create_agent_role(self.name)
        
        # Build the agent configuration
        agent_config = {
            "agentName": self.name,
            "agentResourceRoleArn": role_arn,
            "foundationModel": self.llm,
            "instruction": f"Role: {self.role}\n\nGoal: {self.goal}\n\n{self.instructions}",
            "description": self.goal,
        }
        
        # Add guardrail if specified
        if self.guardrail and self.guardrail.guardrail_id:
            agent_config["guardrailConfiguration"] = {
                "guardrailIdentifier": self.guardrail.guardrail_id,
                "guardrailVersion": self.guardrail.guardrail_version
            }
        
        try:
            # Create the agent
            response = bedrock_agent_client.create_agent(**agent_config)
            self.agent_id = response["agent"]["agentId"]
            self.agent_arn = response["agent"]["agentArn"]
            
            if self.verbose:
                logger.info(f"Created agent with ID: {self.agent_id}")
            
            # Wait for agent to be ready
            self._wait_for_agent_status("NOT_PREPARED")
            
            # Attach tools (action groups)
            for tool in self.tools:
                self._attach_tool(tool)
            
            # Attach knowledge base if specified
            if self.kb_id:
                self._attach_knowledge_base()
            
            # Prepare the agent
            bedrock_agent_client.prepare_agent(agentId=self.agent_id)
            self._wait_for_agent_status("PREPARED")
            
            # Create an alias for invocation
            self._create_alias()
            
            if self.verbose:
                logger.info(f"Agent '{self.name}' is ready for invocation")
                
        except ClientError as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    def _wait_for_agent_status(self, expected_status: str, timeout: int = 120):
        """Wait for the agent to reach the expected status."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = bedrock_agent_client.get_agent(agentId=self.agent_id)
            status = response["agent"]["agentStatus"]
            
            if status == expected_status:
                return
            elif status in ["FAILED", "DELETING"]:
                raise Exception(f"Agent entered unexpected status: {status}")
                
            time.sleep(2)
            
        raise TimeoutError(f"Agent did not reach status {expected_status} within {timeout}s")
    
    def _attach_tool(self, tool: Dict):
        """Attach a tool (action group) to the agent."""
        tool_config = tool.get("definition", {})
        lambda_arn = tool.get("code")
        
        if not lambda_arn or not tool_config:
            logger.warning(f"Skipping invalid tool configuration: {tool}")
            return
        
        # Build the function schema
        function_schema = {
            "functions": [
                {
                    "name": tool_config["name"],
                    "description": tool_config.get("description", ""),
                    "parameters": {
                        k: {
                            "type": v.get("type", "string"),
                            "description": v.get("description", ""),
                            "required": v.get("required", False)
                        }
                        for k, v in tool_config.get("parameters", {}).items()
                    }
                }
            ]
        }
        
        bedrock_agent_client.create_agent_action_group(
            agentId=self.agent_id,
            agentVersion="DRAFT",
            actionGroupName=tool_config["name"],
            actionGroupExecutor={"lambda": lambda_arn},
            functionSchema=function_schema,
            description=tool_config.get("description", "")
        )
        
        if self.verbose:
            logger.info(f"Attached tool '{tool_config['name']}' to agent")
    
    def _attach_knowledge_base(self):
        """Attach a knowledge base to the agent."""
        bedrock_agent_client.associate_agent_knowledge_base(
            agentId=self.agent_id,
            agentVersion="DRAFT",
            knowledgeBaseId=self.kb_id,
            description="Knowledge base for financial document retrieval"
        )
        
        if self.verbose:
            logger.info(f"Attached knowledge base {self.kb_id} to agent")
    
    def _create_alias(self):
        """Create an alias for the agent to enable invocation."""
        response = bedrock_agent_client.create_agent_alias(
            agentId=self.agent_id,
            agentAliasName="live"
        )
        
        self.agent_alias_id = response["agentAlias"]["agentAliasId"]
        self.agent_alias_arn = response["agentAlias"]["agentAliasArn"]
        
        # Wait for alias to be ready
        time.sleep(5)
        
    def invoke(
        self, 
        prompt: str, 
        session_id: Optional[str] = None,
        enable_trace: bool = False,
        trace_level: str = "core"
    ) -> str:
        """
        Invoke the agent with a prompt and get a response.
        
        Args:
            prompt: The user's input/question
            session_id: Optional session ID for conversation continuity
            enable_trace: Whether to enable trace logging
            trace_level: Level of trace detail ("outline", "core", "all")
            
        Returns:
            The agent's response text
        """
        if not self.agent_alias_id:
            raise ValueError("Agent must be deployed before invocation")
        
        session_id = session_id or f"session-{int(time.time())}"
        
        response = bedrock_runtime_client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
            enableTrace=enable_trace
        )
        
        # Stream and collect the response
        result_text = ""
        for event in response["completion"]:
            if "chunk" in event:
                result_text += event["chunk"]["bytes"].decode("utf-8")
                
            # Log trace info if enabled
            if enable_trace and "trace" in event:
                self._log_trace(event["trace"], trace_level)
        
        return result_text
    
    def _log_trace(self, trace: Dict, level: str):
        """Log trace information for debugging."""
        if level == "outline":
            # Just log high-level steps
            if "orchestrationTrace" in trace:
                orch = trace["orchestrationTrace"]
                if "modelInvocationInput" in orch:
                    logger.info("Model invocation started")
                elif "modelInvocationOutput" in orch:
                    logger.info("Model invocation completed")
        elif level in ("core", "all"):
            logger.info(f"Trace: {json.dumps(trace, indent=2, default=str)}")


class SupervisorAgent(Agent):
    """
    A supervisor agent that orchestrates multiple sub-agents.
    
    The supervisor pattern allows a main agent to delegate tasks to
    specialized sub-agents and consolidate their responses into a
    cohesive answer.
    
    Example:
        supervisor = SupervisorAgent.create(
            "research_assistant",
            role="Research Coordinator",
            goal="Orchestrate research tasks",
            collaboration_type="SUPERVISOR",
            collaborator_agents=[
                {"agent": "news_agent", "instructions": "Use for news lookup"},
                {"agent": "data_agent", "instructions": "Use for data analysis"}
            ],
            collaborator_objects=[news_agent, data_agent],
            llm="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        instructions: str,
        llm: str,
        collaboration_type: str = "SUPERVISOR",
        collaborator_agents: Optional[List[Dict]] = None,
        collaborator_objects: Optional[List[Agent]] = None,
        guardrail: Optional[Guardrail] = None,
        verbose: bool = True
    ):
        """
        Initialize a SupervisorAgent instance.
        
        Args:
            name: Unique name for the supervisor
            role: Description of the supervisor's role
            goal: What the supervisor is trying to accomplish
            instructions: Detailed instructions for orchestration
            llm: Model ID to use
            collaboration_type: Type of multi-agent collaboration
            collaborator_agents: List of collaborator configurations
            collaborator_objects: List of actual Agent instances
            guardrail: Optional guardrail for content filtering
            verbose: Whether to log detailed operations
        """
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            llm=llm,
            guardrail=guardrail,
            verbose=verbose
        )
        
        self.collaboration_type = collaboration_type
        self.collaborator_agents = collaborator_agents or []
        self.collaborator_objects = collaborator_objects or []
        
    @classmethod
    def create(
        cls,
        name: str,
        role: str,
        goal: str,
        instructions: str,
        llm: str,
        collaboration_type: str = "SUPERVISOR",
        collaborator_agents: Optional[List[Dict]] = None,
        collaborator_objects: Optional[List[Agent]] = None,
        guardrail: Optional[Guardrail] = None,
        verbose: bool = True
    ) -> "SupervisorAgent":
        """
        Factory method to create and deploy a supervisor agent.
        
        Returns:
            Fully initialized SupervisorAgent ready for orchestration
        """
        supervisor = cls(
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            llm=llm,
            collaboration_type=collaboration_type,
            collaborator_agents=collaborator_agents,
            collaborator_objects=collaborator_objects,
            guardrail=guardrail,
            verbose=verbose
        )
        
        supervisor._deploy()
        return supervisor
    
    def _deploy(self):
        """Deploy the supervisor agent with collaborators."""
        if self.verbose:
            logger.info(f"Creating supervisor agent '{self.name}'...")
            
        # Create IAM role
        role_arn = agents_helper.create_agent_role(self.name)
        
        # Build agent configuration with collaboration settings
        agent_config = {
            "agentName": self.name,
            "agentResourceRoleArn": role_arn,
            "foundationModel": self.llm,
            "instruction": f"Role: {self.role}\n\nGoal: {self.goal}\n\n{self.instructions}",
            "description": self.goal,
            "agentCollaboration": self.collaboration_type,
        }
        
        # Add guardrail if specified
        if self.guardrail and self.guardrail.guardrail_id:
            agent_config["guardrailConfiguration"] = {
                "guardrailIdentifier": self.guardrail.guardrail_id,
                "guardrailVersion": self.guardrail.guardrail_version
            }
        
        try:
            # Create the supervisor agent
            response = bedrock_agent_client.create_agent(**agent_config)
            self.agent_id = response["agent"]["agentId"]
            self.agent_arn = response["agent"]["agentArn"]
            
            if self.verbose:
                logger.info(f"Created supervisor with ID: {self.agent_id}")
            
            # Wait for agent to be ready
            self._wait_for_agent_status("NOT_PREPARED")
            
            # Associate collaborator agents
            for i, collab_config in enumerate(self.collaborator_agents):
                collab_obj = self.collaborator_objects[i] if i < len(self.collaborator_objects) else None
                self._associate_collaborator(collab_config, collab_obj)
            
            # Prepare the agent
            bedrock_agent_client.prepare_agent(agentId=self.agent_id)
            self._wait_for_agent_status("PREPARED")
            
            # Create alias
            self._create_alias()
            
            if self.verbose:
                logger.info(f"Supervisor '{self.name}' is ready for orchestration")
                
        except ClientError as e:
            logger.error(f"Failed to create supervisor agent: {e}")
            raise
    
    def _associate_collaborator(self, config: Dict, agent_obj: Optional[Agent]):
        """Associate a collaborator agent with the supervisor."""
        if not agent_obj or not agent_obj.agent_alias_arn:
            logger.warning(f"Skipping collaborator {config.get('agent')} - not properly initialized")
            return
            
        bedrock_agent_client.associate_agent_collaborator(
            agentId=self.agent_id,
            agentVersion="DRAFT",
            agentDescriptor={"aliasArn": agent_obj.agent_alias_arn},
            collaboratorName=config.get("agent", agent_obj.name),
            collaborationInstruction=config.get("instructions", ""),
            relayConversationHistory="TO_COLLABORATOR"
        )
        
        if self.verbose:
            logger.info(f"Associated collaborator '{config.get('agent')}' with supervisor")
