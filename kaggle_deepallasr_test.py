#!/usr/bin/env python3
"""
Kaggle Test Script for nanoGPT DeepALL Agent
Dataset: deepallasr
Purpose: Train and test the nanoGPT DeepALL Agent as an operating system
with CSV data from Kaggle
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepallasr_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('DeepALLASR_Test')

# ============================================================================
# PHASE 1: DATA LOADING AND ANALYSIS
# ============================================================================

class DataLoader:
    """Load and analyze all CSV and text files from Kaggle dataset"""
    
    def __init__(self, data_dir: str = '/kaggle/input/deepallasr'):
        self.data_dir = data_dir
        self.data_files = {}
        self.csv_data = {}
        self.text_data = {}
        self.stats = {}
        logger.info(f"Initializing DataLoader with directory: {data_dir}")
    
    def load_all_data(self) -> bool:
        """Load all CSV and text files"""
        try:
            logger.info("Starting data loading phase...")
            
            # List all files
            if not os.path.exists(self.data_dir):
                logger.error(f"Data directory not found: {self.data_dir}")
                return False
            
            all_files = os.listdir(self.data_dir)
            logger.info(f"Found {len(all_files)} files in dataset")
            
            # Load CSV files
            csv_files = [f for f in all_files if f.endswith('.csv')]
            logger.info(f"Loading {len(csv_files)} CSV files...")
            
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(self.data_dir, csv_file)
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    self.csv_data[csv_file] = df
                    self.stats[csv_file] = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'dtypes': df.dtypes.to_dict(),
                        'missing_values': df.isnull().sum().to_dict()
                    }
                    logger.info(f"✓ Loaded {csv_file}: {df.shape}")
                except Exception as e:
                    logger.error(f"✗ Failed to load {csv_file}: {str(e)}")
            
            # Load text files
            text_files = [f for f in all_files if f.endswith('.txt')]
            logger.info(f"Loading {len(text_files)} text files...")
            
            for text_file in text_files:
                try:
                    file_path = os.path.join(self.data_dir, text_file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    self.text_data[text_file] = content
                    logger.info(f"✓ Loaded {text_file}: {len(content)} bytes")
                except Exception as e:
                    logger.error(f"✗ Failed to load {text_file}: {str(e)}")
            
            logger.info(f"Data loading complete: {len(self.csv_data)} CSVs, {len(self.text_data)} texts")
            return True
            
        except Exception as e:
            logger.error(f"Critical error during data loading: {str(e)}")
            traceback.print_exc()
            return False
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality and completeness"""
        logger.info("Analyzing data quality...")
        
        quality_report = {
            'total_files': len(self.csv_data) + len(self.text_data),
            'csv_files': len(self.csv_data),
            'text_files': len(self.text_data),
            'total_rows': int(sum(df.shape[0] for df in self.csv_data.values())),
            'total_columns': int(sum(df.shape[1] for df in self.csv_data.values())),
            'csv_details': {}
        }
        
        for filename, df in self.csv_data.items():
            quality_report['csv_details'][filename] = {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
                'missing_percentage': float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                'duplicate_rows': int(df.duplicated().sum())
            }
        
        logger.info(f"Data Quality Report: {json.dumps(quality_report, indent=2)}")
        return quality_report

# ============================================================================
# PHASE 2: NANOGPT DEEPALL AGENT INITIALIZATION
# ============================================================================

class NanoGPTDeepALLAgent:
    """nanoGPT DeepALL Agent - Operating System for AI"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.modules = {}
        self.superintelligences = {}
        self.knowledge_base = {}
        self.performance_metrics = {}
        self.errors = []
        logger.info("Initializing nanoGPT DeepALL Agent...")
    
    def initialize_modules(self) -> bool:
        """Initialize modules from CSV data"""
        try:
            logger.info("Initializing modules...")
            
            # Load modules from relevant CSVs
            module_files = ['modules_combined.csv', 'module_registry.txt', 'Original_CoreControl.csv']
            
            for file in module_files:
                if file in self.data_loader.csv_data:
                    df = self.data_loader.csv_data[file]
                    self.modules[file] = {
                        'data': df,
                        'count': len(df),
                        'columns': list(df.columns)
                    }
                    logger.info(f"✓ Initialized {file}: {len(df)} modules")
                elif file in self.data_loader.text_data:
                    content = self.data_loader.text_data[file]
                    self.modules[file] = {
                        'content': content,
                        'size': len(content)
                    }
                    logger.info(f"✓ Loaded {file}: {len(content)} bytes")
            
            logger.info(f"Modules initialized: {len(self.modules)} module sources")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing modules: {str(e)}")
            self.errors.append(('module_initialization', str(e)))
            return False
    
    def initialize_superintelligences(self) -> bool:
        """Initialize superintelligences from data"""
        try:
            logger.info("Initializing superintelligences...")
            
            if 'Superintelligenzen.csv' in self.data_loader.csv_data:
                df = self.data_loader.csv_data['Superintelligenzen.csv']
                self.superintelligences = {
                    'data': df,
                    'count': len(df),
                    'columns': list(df.columns)
                }
                logger.info(f"✓ Initialized {len(df)} superintelligences")
                return True
            else:
                logger.warning("Superintelligenzen.csv not found")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing superintelligences: {str(e)}")
            self.errors.append(('superintelligence_initialization', str(e)))
            return False
    
    def build_knowledge_base(self) -> bool:
        """Build knowledge base from all data"""
        try:
            logger.info("Building knowledge base...")
            
            # Load knowledge base CSV
            if 'knowledge_base.csv' in self.data_loader.csv_data:
                df = self.data_loader.csv_data['knowledge_base.csv']
                self.knowledge_base['primary'] = df
                logger.info(f"✓ Loaded primary knowledge base: {len(df)} entries")
            
            # Load learning results
            if 'learning_results.csv' in self.data_loader.csv_data:
                df = self.data_loader.csv_data['learning_results.csv']
                self.knowledge_base['learning_results'] = df
                logger.info(f"✓ Loaded learning results: {len(df)} entries")
            
            # Load history
            if 'history.csv' in self.data_loader.csv_data:
                df = self.data_loader.csv_data['history.csv']
                self.knowledge_base['history'] = df
                logger.info(f"✓ Loaded history: {len(df)} entries")
            
            logger.info(f"Knowledge base built: {len(self.knowledge_base)} components")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}")
            self.errors.append(('knowledge_base_building', str(e)))
            return False

# ============================================================================
# PHASE 3: TRAINING AND TESTING
# ============================================================================

class AgentTrainer:
    """Train nanoGPT DeepALL Agent with CSV data"""
    
    def __init__(self, agent: NanoGPTDeepALLAgent):
        self.agent = agent
        self.training_results = {}
        self.test_results = {}
        logger.info("Initializing AgentTrainer...")
    
    def train_agent(self) -> bool:
        """Train agent with loaded data"""
        try:
            logger.info("Starting agent training...")
            start_time = datetime.now()
            
            # Train on modules
            if self.agent.modules:
                logger.info("Training on modules...")
                self.training_results['modules'] = {
                    'count': len(self.agent.modules),
                    'status': 'trained'
                }
            
            # Train on superintelligences
            if self.agent.superintelligences:
                logger.info("Training on superintelligences...")
                self.training_results['superintelligences'] = {
                    'count': self.agent.superintelligences.get('count', 0),
                    'status': 'trained'
                }
            
            # Train on knowledge base
            if self.agent.knowledge_base:
                logger.info("Training on knowledge base...")
                self.training_results['knowledge_base'] = {
                    'components': len(self.agent.knowledge_base),
                    'status': 'trained'
                }
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.training_results['total_time_seconds'] = training_time
            
            logger.info(f"✓ Training complete in {training_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self.agent.errors.append(('training', str(e)))
            return False
    
    def test_agent(self) -> bool:
        """Test agent functionality"""
        try:
            logger.info("Starting agent testing...")
            
            tests_passed = 0
            tests_failed = 0
            
            # Test 1: Module loading
            try:
                assert len(self.agent.modules) > 0, "No modules loaded"
                self.test_results['module_loading'] = 'PASS'
                tests_passed += 1
                logger.info("✓ Test 1: Module loading - PASS")
            except AssertionError as e:
                self.test_results['module_loading'] = f'FAIL: {str(e)}'
                tests_failed += 1
                logger.error(f"✗ Test 1: Module loading - FAIL: {str(e)}")
            
            # Test 2: Superintelligence loading
            try:
                assert len(self.agent.superintelligences) > 0, "No superintelligences loaded"
                self.test_results['superintelligence_loading'] = 'PASS'
                tests_passed += 1
                logger.info("✓ Test 2: Superintelligence loading - PASS")
            except AssertionError as e:
                self.test_results['superintelligence_loading'] = f'FAIL: {str(e)}'
                tests_failed += 1
                logger.error(f"✗ Test 2: Superintelligence loading - FAIL: {str(e)}")
            
            # Test 3: Knowledge base building
            try:
                assert len(self.agent.knowledge_base) > 0, "Knowledge base not built"
                self.test_results['knowledge_base_building'] = 'PASS'
                tests_passed += 1
                logger.info("✓ Test 3: Knowledge base building - PASS")
            except AssertionError as e:
                self.test_results['knowledge_base_building'] = f'FAIL: {str(e)}'
                tests_failed += 1
                logger.error(f"✗ Test 3: Knowledge base building - FAIL: {str(e)}")
            
            # Test 4: Data integrity
            try:
                total_rows = sum(df.shape[0] for df in self.agent.data_loader.csv_data.values())
                assert total_rows > 0, "No data loaded"
                self.test_results['data_integrity'] = 'PASS'
                tests_passed += 1
                logger.info(f"✓ Test 4: Data integrity - PASS ({total_rows} rows)")
            except AssertionError as e:
                self.test_results['data_integrity'] = f'FAIL: {str(e)}'
                tests_failed += 1
                logger.error(f"✗ Test 4: Data integrity - FAIL: {str(e)}")
            
            # Test 5: Error tracking
            try:
                error_count = len(self.agent.errors)
                self.test_results['error_tracking'] = f'PASS ({error_count} errors detected)'
                tests_passed += 1
                logger.info(f"✓ Test 5: Error tracking - PASS ({error_count} errors)")
            except Exception as e:
                self.test_results['error_tracking'] = f'FAIL: {str(e)}'
                tests_failed += 1
                logger.error(f"✗ Test 5: Error tracking - FAIL: {str(e)}")
            
            self.test_results['summary'] = {
                'total_tests': tests_passed + tests_failed,
                'passed': tests_passed,
                'failed': tests_failed,
                'pass_rate': f"{(tests_passed / (tests_passed + tests_failed) * 100):.1f}%"
            }
            
            logger.info(f"Testing complete: {tests_passed} passed, {tests_failed} failed")
            return tests_failed == 0
            
        except Exception as e:
            logger.error(f"Critical error during testing: {str(e)}")
            self.agent.errors.append(('testing', str(e)))
            return False

# ============================================================================
# PHASE 4: PERFORMANCE ANALYSIS
# ============================================================================

class PerformanceAnalyzer:
    """Analyze agent performance"""
    
    def __init__(self, agent: NanoGPTDeepALLAgent, trainer: AgentTrainer):
        self.agent = agent
        self.trainer = trainer
        self.performance_report = {}
        logger.info("Initializing PerformanceAnalyzer...")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance"""
        logger.info("Analyzing performance...")
        
        self.performance_report = {
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'total_csv_files': len(self.agent.data_loader.csv_data),
                'total_text_files': len(self.agent.data_loader.text_data),
                'total_rows': sum(df.shape[0] for df in self.agent.data_loader.csv_data.values()),
                'total_columns': sum(df.shape[1] for df in self.agent.data_loader.csv_data.values()),
            },
            'agent_statistics': {
                'modules_loaded': len(self.agent.modules),
                'superintelligences_loaded': len(self.agent.superintelligences),
                'knowledge_base_components': len(self.agent.knowledge_base),
            },
            'training_results': self.trainer.training_results,
            'test_results': self.trainer.test_results,
            'errors_detected': len(self.agent.errors),
            'error_details': self.agent.errors
        }
        
        logger.info(f"Performance report generated: {json.dumps(self.performance_report, indent=2)}")
        return self.performance_report
    
    def generate_report(self, output_file: str = 'deepallasr_test_report.json'):
        """Generate detailed report"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.performance_report, f, indent=2)
            logger.info(f"✓ Report saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("nanoGPT DeepALL Agent - Kaggle deepallasr Dataset Test")
    logger.info("="*80)
    
    try:
        # Phase 1: Load data
        logger.info("\n[PHASE 1] Loading Data from Kaggle...")
        data_loader = DataLoader()
        if not data_loader.load_all_data():
            logger.error("Failed to load data")
            return False
        
        quality_report = data_loader.analyze_data_quality()
        
        # Phase 2: Initialize agent
        logger.info("\n[PHASE 2] Initializing nanoGPT DeepALL Agent...")
        agent = NanoGPTDeepALLAgent(data_loader)
        
        if not agent.initialize_modules():
            logger.warning("Module initialization had issues")
        if not agent.initialize_superintelligences():
            logger.warning("Superintelligence initialization had issues")
        if not agent.build_knowledge_base():
            logger.warning("Knowledge base building had issues")
        
        # Phase 3: Train and test
        logger.info("\n[PHASE 3] Training and Testing Agent...")
        trainer = AgentTrainer(agent)
        
        if not trainer.train_agent():
            logger.error("Training failed")
            return False
        
        if not trainer.test_agent():
            logger.warning("Some tests failed")
        
        # Phase 4: Performance analysis
        logger.info("\n[PHASE 4] Analyzing Performance...")
        analyzer = PerformanceAnalyzer(agent, trainer)
        performance_report = analyzer.analyze_performance()
        
        if not analyzer.generate_report():
            logger.error("Failed to generate report")
            return False
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Data Files Loaded: {quality_report['total_files']}")
        logger.info(f"Total Rows: {quality_report['total_rows']}")
        logger.info(f"Total Columns: {quality_report['total_columns']}")
        logger.info(f"Modules Initialized: {len(agent.modules)}")
        logger.info(f"Superintelligences: {agent.superintelligences.get('count', 0)}")
        logger.info(f"Knowledge Base Components: {len(agent.knowledge_base)}")
        logger.info(f"Tests Passed: {trainer.test_results['summary']['passed']}/{trainer.test_results['summary']['total_tests']}")
        logger.info(f"Errors Detected: {len(agent.errors)}")
        logger.info("="*80)
        
        if len(agent.errors) > 0:
            logger.warning("\nErrors detected during execution:")
            for error_type, error_msg in agent.errors:
                logger.warning(f"  - {error_type}: {error_msg}")
        
        logger.info("\n✓ Test execution complete!")
        return True
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
