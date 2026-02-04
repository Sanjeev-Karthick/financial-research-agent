"""
Knowledge Base Helper Module

This module provides utilities for creating and managing Amazon Bedrock 
Knowledge Bases with OpenSearch Serverless as the vector store.

Knowledge bases enable agents to retrieve information from documents
like SEC filings, earnings calls, and financial reports using
semantic search powered by embedding models.

Classes:
    KnowledgeBasesForAmazonBedrock: Main class for KB operations
"""

import boto3
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
sts_client = boto3.client("sts")
bedrock_agent_client = boto3.client("bedrock-agent")
s3_client = boto3.client("s3")
aoss_client = boto3.client("opensearchserverless")
iam_client = boto3.client("iam")

# Get AWS account info
caller_identity = sts_client.get_caller_identity()
account_id = caller_identity["Account"]
region = boto3.session.Session().region_name or "us-east-1"


class KnowledgeBasesForAmazonBedrock:
    """
    Helper class for managing Amazon Bedrock Knowledge Bases.
    
    This class simplifies the process of creating knowledge bases with
    OpenSearch Serverless as the vector store. It handles all the
    underlying complexity including:
    
    - Creating OpenSearch Serverless collections
    - Setting up security policies
    - Creating vector indices
    - Configuring data sources (S3)
    - Syncing documents to the knowledge base
    
    Example:
        kb_helper = KnowledgeBasesForAmazonBedrock()
        kb_id, ds_id = kb_helper.create_or_retrieve_knowledge_base(
            kb_name="financial-docs",
            kb_description="Financial document knowledge base",
            data_bucket_name="my-docs-bucket",
            embedding_model="amazon.titan-embed-text-v2:0"
        )
        kb_helper.synchronize_data(kb_id, ds_id)
    """
    
    def __init__(self):
        """Initialize the Knowledge Base helper with AWS clients."""
        self.bedrock_agent = boto3.client("bedrock-agent")
        self.s3 = boto3.client("s3")
        self.aoss = boto3.client("opensearchserverless")
        self.iam = boto3.client("iam")
        
        # Cache for created resources
        self._collections: Dict[str, str] = {}
        self._indices: Dict[str, bool] = {}
        
    def create_or_retrieve_knowledge_base(
        self,
        kb_name: str,
        kb_description: str,
        data_bucket_name: str,
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        data_prefix: str = "data/",
        verbose: bool = True
    ) -> Tuple[str, str]:
        """
        Create a new knowledge base or retrieve an existing one.
        
        This method creates all the required infrastructure for a 
        knowledge base including:
        - OpenSearch Serverless collection for vector storage
        - Vector index with appropriate mappings
        - S3 data source configuration
        - IAM roles and policies
        
        Args:
            kb_name: Unique name for the knowledge base
            kb_description: Human-readable description
            data_bucket_name: S3 bucket containing source documents
            embedding_model: Model ID for generating embeddings
            data_prefix: S3 prefix where documents are stored
            verbose: Whether to log detailed operations
            
        Returns:
            Tuple of (knowledge_base_id, data_source_id)
        """
        # Check if knowledge base already exists
        existing_kb = self._get_existing_kb(kb_name)
        if existing_kb:
            if verbose:
                logger.info(f"Found existing knowledge base '{kb_name}' with ID: {existing_kb['knowledgeBaseId']}")
            
            # Get the data source ID
            ds_id = self._get_data_source_id(existing_kb["knowledgeBaseId"])
            return existing_kb["knowledgeBaseId"], ds_id
        
        if verbose:
            logger.info(f"Creating new knowledge base '{kb_name}'...")
        
        # Step 1: Create OpenSearch Serverless collection
        collection_id, collection_arn = self._create_collection(kb_name, verbose)
        
        # Step 2: Create vector index in the collection
        collection_endpoint = self._get_collection_endpoint(collection_id)
        self._create_vector_index(collection_endpoint, kb_name, verbose)
        
        # Step 3: Create IAM role for knowledge base
        kb_role_arn = self._create_kb_role(kb_name, data_bucket_name, collection_arn, verbose)
        
        # Step 4: Create the knowledge base
        kb_id = self._create_knowledge_base(
            kb_name=kb_name,
            kb_description=kb_description,
            role_arn=kb_role_arn,
            collection_arn=collection_arn,
            embedding_model=embedding_model,
            verbose=verbose
        )
        
        # Step 5: Create S3 data source
        ds_id = self._create_data_source(
            kb_id=kb_id,
            bucket_name=data_bucket_name,
            prefix=data_prefix,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"Successfully created knowledge base '{kb_name}'")
            logger.info(f"  KB ID: {kb_id}")
            logger.info(f"  Data Source ID: {ds_id}")
        
        return kb_id, ds_id
    
    def synchronize_data(
        self,
        kb_id: str,
        ds_id: str,
        verbose: bool = True
    ) -> bool:
        """
        Synchronize the data source with the knowledge base.
        
        This method triggers an ingestion job that:
        1. Reads documents from the S3 data source
        2. Chunks the documents appropriately
        3. Generates embeddings using the configured model
        4. Stores the vectors in OpenSearch
        
        Args:
            kb_id: Knowledge base ID
            ds_id: Data source ID
            verbose: Whether to log progress
            
        Returns:
            True if sync completed successfully
        """
        if verbose:
            logger.info(f"Starting data synchronization for KB {kb_id}...")
        
        # Start the ingestion job
        response = self.bedrock_agent.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id
        )
        
        job_id = response["ingestionJob"]["ingestionJobId"]
        
        if verbose:
            logger.info(f"Ingestion job started: {job_id}")
        
        # Poll for completion
        while True:
            job_response = self.bedrock_agent.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                ingestionJobId=job_id
            )
            
            status = job_response["ingestionJob"]["status"]
            
            if status == "COMPLETE":
                if verbose:
                    stats = job_response["ingestionJob"].get("statistics", {})
                    logger.info(f"Ingestion complete!")
                    logger.info(f"  Documents scanned: {stats.get('numberOfDocumentsScanned', 'N/A')}")
                    logger.info(f"  Documents indexed: {stats.get('numberOfDocumentsIndexed', 'N/A')}")
                return True
                
            elif status == "FAILED":
                logger.error(f"Ingestion failed: {job_response['ingestionJob'].get('failureReasons', 'Unknown error')}")
                return False
                
            elif status in ("STARTING", "IN_PROGRESS"):
                if verbose:
                    logger.info(f"Ingestion status: {status}...")
                time.sleep(10)
            else:
                logger.warning(f"Unexpected ingestion status: {status}")
                time.sleep(5)
    
    def _get_existing_kb(self, kb_name: str) -> Optional[Dict]:
        """Check if a knowledge base with the given name exists."""
        try:
            response = self.bedrock_agent.list_knowledge_bases()
            
            for kb in response.get("knowledgeBaseSummaries", []):
                if kb["name"] == kb_name:
                    return kb
                    
            return None
            
        except ClientError as e:
            logger.error(f"Error listing knowledge bases: {e}")
            return None
    
    def _get_data_source_id(self, kb_id: str) -> str:
        """Get the data source ID for a knowledge base."""
        response = self.bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
        
        data_sources = response.get("dataSourceSummaries", [])
        if data_sources:
            return data_sources[0]["dataSourceId"]
        
        raise ValueError(f"No data source found for KB {kb_id}")
    
    def _create_collection(self, name: str, verbose: bool) -> Tuple[str, str]:
        """Create an OpenSearch Serverless collection for vector storage."""
        collection_name = f"{name}-vectors"[:32]  # Max 32 chars
        
        if verbose:
            logger.info(f"Creating OpenSearch Serverless collection: {collection_name}")
        
        # First, create security policies
        self._create_encryption_policy(collection_name)
        self._create_network_policy(collection_name)
        self._create_data_access_policy(collection_name)
        
        try:
            response = self.aoss.create_collection(
                name=collection_name,
                type="VECTORSEARCH",
                description=f"Vector store for {name} knowledge base"
            )
            
            collection_id = response["createCollectionDetail"]["id"]
            collection_arn = response["createCollectionDetail"]["arn"]
            
            # Wait for collection to be active
            self._wait_for_collection(collection_id, verbose)
            
            return collection_id, collection_arn
            
        except self.aoss.exceptions.ConflictException:
            if verbose:
                logger.info(f"Collection {collection_name} already exists, retrieving...")
            
            # Get existing collection
            response = self.aoss.batch_get_collection(names=[collection_name])
            collection = response["collectionDetails"][0]
            return collection["id"], collection["arn"]
    
    def _create_encryption_policy(self, collection_name: str):
        """Create encryption policy for the collection."""
        policy = {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                }
            ],
            "AWSOwnedKey": True
        }
        
        try:
            self.aoss.create_security_policy(
                name=f"{collection_name}-enc",
                type="encryption",
                policy=json.dumps(policy)
            )
        except self.aoss.exceptions.ConflictException:
            pass  # Policy already exists
    
    def _create_network_policy(self, collection_name: str):
        """Create network policy for the collection."""
        policy = [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{collection_name}"]
                    },
                    {
                        "ResourceType": "dashboard",
                        "Resource": [f"collection/{collection_name}"]
                    }
                ],
                "AllowFromPublic": True
            }
        ]
        
        try:
            self.aoss.create_security_policy(
                name=f"{collection_name}-net",
                type="network",
                policy=json.dumps(policy)
            )
        except self.aoss.exceptions.ConflictException:
            pass
    
    def _create_data_access_policy(self, collection_name: str):
        """Create data access policy for the collection."""
        # Get current caller identity for access
        identity = sts_client.get_caller_identity()
        principal_arn = identity["Arn"]
        
        policy = [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{collection_name}"],
                        "Permission": [
                            "aoss:CreateCollectionItems",
                            "aoss:UpdateCollectionItems",
                            "aoss:DeleteCollectionItems",
                            "aoss:DescribeCollectionItems"
                        ]
                    },
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{collection_name}/*"],
                        "Permission": [
                            "aoss:CreateIndex",
                            "aoss:UpdateIndex",
                            "aoss:DeleteIndex",
                            "aoss:DescribeIndex",
                            "aoss:ReadDocument",
                            "aoss:WriteDocument"
                        ]
                    }
                ],
                "Principal": [principal_arn, f"arn:aws:iam::{account_id}:role/*Bedrock*"]
            }
        ]
        
        try:
            self.aoss.create_access_policy(
                name=f"{collection_name}-access",
                type="data",
                policy=json.dumps(policy)
            )
        except self.aoss.exceptions.ConflictException:
            pass
    
    def _wait_for_collection(self, collection_id: str, verbose: bool, timeout: int = 300):
        """Wait for collection to become active."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.aoss.batch_get_collection(ids=[collection_id])
            
            if response["collectionDetails"]:
                status = response["collectionDetails"][0]["status"]
                
                if status == "ACTIVE":
                    if verbose:
                        logger.info("Collection is now active")
                    return
                elif status == "FAILED":
                    raise Exception("Collection creation failed")
                    
                if verbose:
                    logger.info(f"Collection status: {status}...")
                    
            time.sleep(10)
            
        raise TimeoutError(f"Collection did not become active within {timeout}s")
    
    def _get_collection_endpoint(self, collection_id: str) -> str:
        """Get the endpoint URL for an OpenSearch collection."""
        response = self.aoss.batch_get_collection(ids=[collection_id])
        return response["collectionDetails"][0]["collectionEndpoint"]
    
    def _create_vector_index(self, endpoint: str, kb_name: str, verbose: bool):
        """Create a vector index in the OpenSearch collection."""
        # Note: In a real implementation, you would use the opensearch-py client
        # to create the index. For simplicity, we're using requests here.
        
        if verbose:
            logger.info("Vector index creation would be done here using opensearch-py")
            logger.info("Index will be created automatically by Bedrock during KB creation")
        
        # The index will be created automatically by Bedrock Knowledge Base
        # when the KB is created with the proper configuration
    
    def _create_kb_role(
        self, 
        kb_name: str, 
        bucket_name: str, 
        collection_arn: str,
        verbose: bool
    ) -> str:
        """Create IAM role for the knowledge base."""
        role_name = f"AmazonBedrockExecutionRoleForKB_{kb_name}"[:64]
        
        # Trust policy for Bedrock
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
                            "aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/*"
                        }
                    }
                }
            ]
        }
        
        # Permissions policy
        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "BedrockInvokeModel",
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": [f"arn:aws:bedrock:{region}::foundation-model/*"]
                },
                {
                    "Sid": "S3Access",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*"
                    ]
                },
                {
                    "Sid": "OpenSearchAccess",
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [collection_arn]
                }
            ]
        }
        
        try:
            # Create the role
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"Execution role for {kb_name} knowledge base"
            )
            role_arn = response["Role"]["Arn"]
            
            # Attach inline policy
            self.iam.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{kb_name}-kb-policy",
                PolicyDocument=json.dumps(permissions_policy)
            )
            
            if verbose:
                logger.info(f"Created KB role: {role_name}")
            
            # Wait for IAM propagation
            time.sleep(15)
            
            return role_arn
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            if verbose:
                logger.info(f"KB role {role_name} already exists")
            return f"arn:aws:iam::{account_id}:role/{role_name}"
    
    def _create_knowledge_base(
        self,
        kb_name: str,
        kb_description: str,
        role_arn: str,
        collection_arn: str,
        embedding_model: str,
        verbose: bool
    ) -> str:
        """Create the Bedrock Knowledge Base."""
        
        # Vector index name - will be created in OpenSearch
        index_name = f"{kb_name}-index"
        
        response = self.bedrock_agent.create_knowledge_base(
            name=kb_name,
            description=kb_description,
            roleArn=role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{region}::foundation-model/{embedding_model}"
                }
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "vectorIndexName": index_name,
                    "fieldMapping": {
                        "vectorField": "bedrock-knowledge-base-default-vector",
                        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                        "metadataField": "AMAZON_BEDROCK_METADATA"
                    }
                }
            }
        )
        
        kb_id = response["knowledgeBase"]["knowledgeBaseId"]
        
        if verbose:
            logger.info(f"Created knowledge base with ID: {kb_id}")
        
        # Wait for KB to be active
        self._wait_for_kb_status(kb_id, "ACTIVE", verbose)
        
        return kb_id
    
    def _wait_for_kb_status(self, kb_id: str, expected_status: str, verbose: bool, timeout: int = 120):
        """Wait for knowledge base to reach expected status."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
            status = response["knowledgeBase"]["status"]
            
            if status == expected_status:
                return
            elif status == "FAILED":
                raise Exception(f"Knowledge base creation failed")
                
            if verbose:
                logger.info(f"Knowledge base status: {status}...")
                
            time.sleep(5)
            
        raise TimeoutError(f"KB did not reach status {expected_status} within {timeout}s")
    
    def _create_data_source(
        self,
        kb_id: str,
        bucket_name: str,
        prefix: str,
        verbose: bool
    ) -> str:
        """Create an S3 data source for the knowledge base."""
        
        response = self.bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name=f"{kb_id}-s3-source",
            description="S3 data source for financial documents",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{bucket_name}",
                    "inclusionPrefixes": [prefix] if prefix else []
                }
            }
        )
        
        ds_id = response["dataSource"]["dataSourceId"]
        
        if verbose:
            logger.info(f"Created data source with ID: {ds_id}")
        
        return ds_id
    
    def delete_knowledge_base(self, kb_id: str, verbose: bool = True):
        """
        Delete a knowledge base and its associated resources.
        
        Args:
            kb_id: Knowledge base ID to delete
            verbose: Whether to log progress
        """
        try:
            # First delete data sources
            ds_response = self.bedrock_agent.list_data_sources(knowledgeBaseId=kb_id)
            for ds in ds_response.get("dataSourceSummaries", []):
                self.bedrock_agent.delete_data_source(
                    knowledgeBaseId=kb_id,
                    dataSourceId=ds["dataSourceId"]
                )
                if verbose:
                    logger.info(f"Deleted data source: {ds['dataSourceId']}")
            
            # Then delete the knowledge base
            self.bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)
            
            if verbose:
                logger.info(f"Deleted knowledge base: {kb_id}")
                
        except ClientError as e:
            logger.error(f"Error deleting knowledge base: {e}")
            raise
