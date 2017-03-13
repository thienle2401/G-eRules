/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package moa.classifiers.rules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import moa.classifiers.AbstractClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.streams.generators.RandomTreeGenerator;

public class GeRules extends AbstractClassifier {
    
	// option for inital sliding size
	public IntOption slidingWindowsSizeOption = new IntOption(
			"slidingWindowsSize", 
			's', 
			"The number of instances in each window", 
		500, 0, Integer.MAX_VALUE);
	
	// option for minmum of tries before a rule is considered to be removed
	public IntOption MinRuleTriesOption = new IntOption(
			"minRuleTries", 
			'm', 
			"Minimum number of rules before it is considered for validation", 
		10, 2, Integer.MAX_VALUE);
	
	// threshold for the rule to be removed
	public FloatOption ruleValidationThresholdOption = new FloatOption("ruleValidationThreshold", 't', "If the accuracy of rule drops below this threshold then the rule will be removed", 0.8d);
	
	
	@Override
	public String getPurposeString() {
		// TODO Auto-generated method stub
		return "Hoeffding Rules classifer by Thien Le, Univeristy of Reading, Uk";
	}

	// store classification distribution throughout the stream
	DoubleVector observedClassDistribution;
	
	// sliding windows buffer for instances
	ArrayList<Instance> slidingWindowsBuffer;
	
	// this buffer store unlearnt instances (if a batch contains instances from a class only then PRISM won't induce any rules)
	ArrayList<Instance> unlearntInstancesList;
	
	// main classifier based on Prism to induce Rules as in the paper
	PrismClassifier prismClassifier;
	
	// rules library induced by the classifer throughout the stream
	List<Rule> rulesList;
	
	// total seen instance
	int totalSeenInstances;
	
	int actualAttempts;
	int actualAttemptsCorrectlyClassified;
	
	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		
		// increase no. of seen intances
		totalSeenInstances++;
		
		// check if there is any rules that cover the instance
		ArrayList<Rule> coveredRules = RulesCoveredInstance(inst);
//		logger.debug("No. Rules cover instance: " + coveredRules.size());
		
//		logger.debug(inst);
		// return prediction if there are rules that cover the instance
		if(coveredRules.size() > 0){
			
			actualAttempts++;
			
			double[] classPrediction = new double[inst.numClasses()];
			// vote class labels from all available rules
			
			for (Rule rule : coveredRules) {
				classPrediction[(int)rule.classification]++;
//				logger.debug(rule.printRule());
                        }
                        
			// actual attempt
			if(Utils.maxIndex(classPrediction) == (int) inst.classValue()){
				actualAttemptsCorrectlyClassified++;
			}
			return classPrediction ;
		}
		
		// otherwise, return the majority class
		return observedClassDistribution.getArrayCopy();
	}
	
	// abstaining rate
	public double abstainingRate(){
		
		double abstainingRate = (double) actualAttempts / (double) totalSeenInstances;
		return abstainingRate;
	}
	
	// tentative accuracy
	public double tentativeAccuracy(){
		
		double tentativeAccuracy = (double) actualAttemptsCorrectlyClassified / (double) actualAttempts;
		return tentativeAccuracy;
	}
	
	@Override
	public void resetLearningImpl() {
		// TODO Auto-generated method stub
		
		// initialise varibles
		prismClassifier = new PrismClassifier();
		rulesList = new ArrayList<>();
		observedClassDistribution = new DoubleVector();
		totalSeenInstances = 0;
		slidingWindowsBuffer = new ArrayList<>(); // not to initialise proper instances without header instance from the data stream
		unlearntInstancesList = new ArrayList<>();
	
		actualAttempts = 0;
		actualAttemptsCorrectlyClassified = 0;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Auto-generated method stub
		
		// add weight of respective class to classification distribution
		observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());
		
		// only add instances to be learnt if there are no rule coverd the instance
		if(RulesCoveredInstance(inst).isEmpty()){
			slidingWindowsBuffer.add(inst);
		}
		// if there are rule(s) cover the instance, then update stattic in the rule
		else{
			
			// for each rule matched the instance,
			// update class distribution statistic
			for (Rule rule : RulesCoveredInstance(inst)) {
				rule.updateClassDistribution(inst);
				
				rule.noOfCovered++;
				
				// also update if the rule correctly cover an instance with it class
				if(inst.classValue() == rule.classification){
					rule.noOfCorrectlyCovered++;
				}else{					// validate the current rule
					if(rule.ruleShouldBeRemoved()){
						rulesList.remove(rule);
					}	
				}	
			}
			
		}
		
		// check if the sliding windows buffer is filled to the criteria
		if(slidingWindowsBuffer.size() == slidingWindowsSizeOption.getValue()){
						
                    // learn rules with the classifier
                    ArrayList<Rule> learntRules = prismClassifier.learnRules(slidingWindowsBuffer);

                    if(learntRules != null){
                            rulesList.addAll(learntRules);
                    }

                    // clear sliding window buffer to take more instances
                    slidingWindowsBuffer.clear();
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		
		return new Measurement[]{
                new Measurement("Abstaining Rate", abstainingRate()),
                new Measurement("Tentative Accuracy", tentativeAccuracy())};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}
	
	public void learnRules(ArrayList<Instance> instances){
		PrismClassifier prismClassifier = new PrismClassifier();
		prismClassifier.learnRules(instances);
	}
	
	public ArrayList<Rule> RulesCoveredInstance(Instance instance){
		
		ArrayList<Rule> coveredRule = new ArrayList<>();
		
		for (Rule rule : rulesList) {
			if(rule.coveredByRule(instance)){
				coveredRule.add(rule);
			}
		}
		return coveredRule;
	}
	
	public class PrismClassifier{
		
		public ArrayList<Rule> learnRules(ArrayList<Instance> instancesListIn){
			
			// set training set from input set
			ArrayList<Instance> inputInstances = new ArrayList<>(instancesListIn);
                        
			// check if there are instances from unlearnt buffer
			if(unlearntInstancesList.size() > 0){
				inputInstances.addAll(unlearntInstancesList);
				unlearntInstancesList.clear();
			}
			
			// work out number of class labels
			 int noClassLabels = inputInstances.get(0).numClasses();
			
			// list of attributes for the data instance
			ArrayList<Attribute> attributesList = Collections.list(inputInstances.get(0).enumerateAttributes());
						
						
			// Original datasets || need to be reset when move on to new classification
			ArrayList<Instance> originalDataset = new ArrayList<>(inputInstances);
			
			// rules library to store 
			ArrayList<Rule> rulesList = new ArrayList<>();
			
			// dataset inormation instance
			Instance infoInstance = (Instance) inputInstances.get(0).copy();
			
			// check if the dataset containt more than one class
			if(noOfClassifications(inputInstances) <= 1){
				unlearntInstancesList.addAll(inputInstances);
				return null;
			}
			
			// Map of Gauss distributions for attribute
			Map<Attribute, Map<Double, NormalDistribution>> numericAttributeClassGaussDistributions = new HashMap<>(initialiseGaussianDistributionForNumericAttribute(infoInstance, inputInstances));
		
			
			
			// loop through each class label
			for (double classLabelIndex = 0; classLabelIndex < noClassLabels;) {
				
				// D dataset for the classification
				ArrayList<Instance> datasetD = new ArrayList<>(originalDataset);
				
				// probabilities for each attribute-value
				TreeMap<Double, RuleTerm> attributeValuesProbability = new TreeMap<>();
				
				// used attribute, and should not be used to induce new rule term
				List<Attribute>	usedAttributes = new ArrayList<>();
				
				// An empty rule
				Rule rule = new Rule(
						(int) noClassLabels,
						classLabelIndex, 
						infoInstance.classAttribute(), 
						GeRules.this.totalSeenInstances);
				boolean goodRule = true;
				
				while(containOtherClasses(datasetD,classLabelIndex) != true){
										
					// Calcuate probability of occurences for each attribute-value
					for (Attribute attribute : attributesList) {
						
						// stop if this attribute is already used to induce rule term
						if(usedAttributes.contains(attribute) == false){
							// check whether the attribute is categorical or numeric
							if(attribute.isNominal()){
								
								for (double i = 0; i < attribute.numValues(); i++) {
									
									// create RuleTerm for this attribute-value
									RuleTerm ruleTerm = new RuleTerm(attribute, i);
									attributeValuesProbability.put(
											calculateProbabilityOfOccurence(ruleTerm, datasetD, classLabelIndex),
											ruleTerm);
								}
								
							// the attribute is numeric
							}else{
								
								NormalDistribution normalDistribution = null;
								normalDistribution = numericAttributeClassGaussDistributions.get(attribute).get(classLabelIndex);
								
								
								
								double[] bounds = findLowerUpperNumericAttributeForClassLabel(normalDistribution, attribute, datasetD);
								double rangeProbability = normalDistribution.probability(bounds[0], bounds[1]);
								
								RuleTerm ruleTerm = new RuleTerm(attribute, bounds[0], bounds[1]);
								
								attributeValuesProbability.put(rangeProbability, ruleTerm);
							}				
						}
					}	
					
					// check class clash
					if(attributeValuesProbability.isEmpty()){
						if(isClassTheMajortiy(datasetD, classLabelIndex)){
							// stop inducing new rule term for current dataset
							break;
						}else{
							
							// don't include this rule in rule lib
							goodRule = false;
							break;
						}
					}
					
					// selecting the ruleTerm (attribute-value) with highest conditional probability
					// no need to induces ruleterm and the rule term probability = 0 (useless)
					if(attributeValuesProbability.lastEntry().getKey() == 0.0d){
						break;
					}
					RuleTerm selectedRuleTerm = attributeValuesProbability.lastEntry().getValue();
					// add ruleTerm to rule
					rule.addRuleTerm(selectedRuleTerm);
					
					// add used attribute
					usedAttributes.add(selectedRuleTerm.attribute);
					
//					logger.debug("Adding rule term to rule");
					
					// creating subset of datasets, containing all instances covered by selected ruleTerm					
					datasetD = instancesCoveredByRuleTerm(datasetD, selectedRuleTerm);
					
					// clear attribute probability
					attributeValuesProbability.clear();
				}
				
				// add rule to ruleslibrary
				if(goodRule && rule.listOfRuleTerm.size() > 0){
				
					// no. of instances covered by rule at once complete for statictic monitor
					rule.setInstancesCoveredWhenRuleCreated(datasetD.size());
					rulesList.add(rule);
				}
				
				// removed instances covered by rule from original datasets
				originalDataset = removeInstances(originalDataset, datasetD);
								
				// check no more instances from given class
				if(notContainClassification(datasetD, classLabelIndex)){
					
					classLabelIndex++;
					
					// reset original dataset for new classification
					originalDataset.clear();
					originalDataset = new ArrayList<>(inputInstances);
					
				}
			}
			return rulesList;
		}
		
		private void initPrism(){
			
		}
		
		private double calculateProbabilityOfOccurence(RuleTerm ruleTerm, ArrayList<Instance> dataset, double classification ){
			
			int totalNoInstancesCoveredByRuleTerm = 0;
			int totalNoInstancesCoveredByRuleTermWithClassification = 0;
			List<Instance> instancesList = new ArrayList<>(dataset);
			
			for (Instance instance : instancesList) {
                            
				if(instance.value(ruleTerm.attribute) == ruleTerm.value){

					totalNoInstancesCoveredByRuleTerm++;
					
					// check if ruleTerm covered instance for specific classification
					if(instance.classValue() == classification){
						totalNoInstancesCoveredByRuleTermWithClassification++;
					}
					
				}
			}
			
			double probabilityOfOccurencesForClassification = (double) totalNoInstancesCoveredByRuleTermWithClassification / (double) totalNoInstancesCoveredByRuleTerm;
			
			
			return Double.isNaN(probabilityOfOccurencesForClassification) ? 0.0d : probabilityOfOccurencesForClassification;
		}
		
		private ArrayList<Instance> instancesCoveredByRuleTerm(ArrayList<Instance> instances, RuleTerm ruleTerm){
			
			List<Instance> instancesList = new ArrayList<>(instances);
			List<Instance> instancesCoveredList = new ArrayList<>();
			
			for (Instance instance : instancesList) {
				if(ruleTerm.coveredByRuleTerm(instance)){					
					instancesCoveredList.add(instance);
				}
			}
			
			
			return (ArrayList<Instance>) instancesCoveredList;
		}
		
		private boolean containOtherClasses(ArrayList<Instance> instances, double classification){
			
			List<Instance> instancesList = new ArrayList<>(instances);
			
			for (Instance instance : instancesList) {
				if(instance.classValue() != classification){
					return false;
				}
			}
			return true;
		}
		
		private ArrayList<Instance>  removeInstances(ArrayList<Instance> instancesA, ArrayList<Instance> instancesB){
			
			ArrayList<Instance> afterRemovedInstancesList = new ArrayList<>();
			
			InstanceComparator instanceComparator = new InstanceComparator(true);
			
			List<Instance> instancesAList = new ArrayList<>(instancesA);
			List<Instance> instancesBList = new ArrayList<>(instancesB);
			
			List<Integer> indexInstancesToBeRemoved = new ArrayList<>();
			
			int i = 0;			
			
			for (Instance instance : instancesAList) {
				
				for (Instance instancej : instancesBList) {
				
					if(instanceComparator.compare(instance, instancej) == 0){
						indexInstancesToBeRemoved.add(i);
						break;
					}
				}
				i++;
			}
			
			// Sort the index to be removed
			Collections.sort(indexInstancesToBeRemoved);
			
			// removing all instances of B from A
			// reverse iteration is used to avoid order mess up in ArrayList
			for (int j2 = indexInstancesToBeRemoved.size() - 1; j2 >= 0; j2--) {
				
				// need to cast to int for removing at index
				instancesAList.remove((int)indexInstancesToBeRemoved.get(j2));
			}
			
			for (Instance instance : instancesAList) {
				afterRemovedInstancesList.add(instance);
			}
						
			return afterRemovedInstancesList;
		}
		
		private boolean notContainClassification(ArrayList<Instance> instances, double classification){
			
			List<Instance> instancesList = new ArrayList<>(instances);
			
			for (Instance instance : instancesList) {
				if(instance.classValue() == classification){
					
					return false;
				}
			}
			
			return true;
		}
		
		// check if given class is majority in given dataset
		private boolean isClassTheMajortiy(ArrayList<Instance> instances, double classification){
			
			List<Instance> instancesList = new ArrayList<>(instances);
			TreeMap<Double, Double> classificationProbability = new TreeMap<>();
			Attribute classAttribute = instances.get(0).classAttribute();
			
			for (double i = 0; i < classAttribute.numValues(); i++) {
				int matchedClassCount = 0;
				
				for (Instance instance : instancesList) {
					if(instance.classValue() == i){
						matchedClassCount++;
					}
				}
				
				classificationProbability.put(((double) matchedClassCount / (double) instancesList.size()), i);
			}
			
			return (classificationProbability.lastEntry().getValue() == classification);
		}
		
		// return no. of class labels in the dataset
		private int noOfClassifications(ArrayList<Instance> instances){
			Set<Integer> classLabels = new LinkedHashSet<>();
			
			for (Instance instance : instances) {
				// add class label to the list if not already exist
				if(classLabels.contains((int)instance.classValue()) == false){
					classLabels.add((int)instance.classValue());
				}
			}
			
			return classLabels.size();
		}
		
		private Map<Attribute, Map<Double, NormalDistribution>> initialiseGaussianDistributionForNumericAttribute(Instance instanceInfo, ArrayList<Instance> instancesList){
			
			Map<Attribute, Map<Double, NormalDistribution>> numericAttributeClassGaussDistributions = new HashMap<>();
			
			// go through each numeric attibute
			for (Attribute attribute : Collections.list(instanceInfo.enumerateAttributes())) {
				
				// check whether the attribute is numeric
				if(attribute.isNumeric()){
					
					// for each class label
					HashMap<Double, NormalDistribution> classLabelDistribution = new HashMap<>();
					for (int classLabelNo = 0; classLabelNo < instanceInfo.numClasses(); classLabelNo++) {
						
						// go through all instance in the dataset to create normal distribution
						SummaryStatistics summaryStatistics = new SummaryStatistics();
						for (Instance instance : instancesList) {
							
							summaryStatistics.addValue(instance.value(attribute));
						}
						
						// create normal distribution for this attribute with corresponding
						// class label
						NormalDistribution normalDistribution = new NormalDistribution(
								summaryStatistics.getMean(), 
								summaryStatistics.getStandardDeviation());
						
						// map to hold classLabel and distribution
						classLabelDistribution.put((double) classLabelNo, normalDistribution);
						
					}
					
					// put it into the map
					numericAttributeClassGaussDistributions.put(attribute, classLabelDistribution);
				}
				
			}
						
			return numericAttributeClassGaussDistributions;
		}
		
		private double calculateDensityProbabilityNumericAttribute(RuleTerm ruleTerm, NormalDistribution normalDistribution){
			
			// Cumulative probability for given class label
			double DensityProbability = normalDistribution.density(ruleTerm.value);
			
			return DensityProbability;
		}
		
		//Hoeffding Bound 
		private double ComputeHoeffdingBound(double range, double confidence,
			            int n) {
			return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
			                / (2.0 * n));

		}
		
		// work out lowerbound and upperbound from dataset for given 
		// normal distribution of numeric attribute
		private double[] findLowerUpperNumericAttributeForClassLabel(NormalDistribution normalDistribution, Attribute attribute, ArrayList<Instance> instancesList){
			
			// density probaility for each numeric value from the attribute from the dataset
			TreeMap<Double, Double>	densityProbabilitiesOfValues = new TreeMap<>();
			
			for (Instance instance : instancesList) {
				
				// get value of the attribute from an instance
				double valueOfAttribute = instance.value(attribute);
				
				// calculate density probability
				double densityProbability = normalDistribution.density(valueOfAttribute);
				
				// add to density probability map to find the best later
				densityProbabilitiesOfValues.put(densityProbability, valueOfAttribute);
				
			}
			
			double valueWithBestDensityProbability = densityProbabilitiesOfValues.lastEntry().getValue();
			
			TreeMap<Double, Double> smallerSideOfValueWithBestProbability = new TreeMap<>();
			TreeMap<Double, Double> greaterSideOfValueWithBestProbability = new TreeMap<>();

			// work out list of values and their corresponding density probability
			for (Entry<Double, Double> probabilityValueEntry : densityProbabilitiesOfValues.entrySet()) {
				
				// value greater than best value should be added to greater side
				if(probabilityValueEntry.getValue() > valueWithBestDensityProbability){
					
					greaterSideOfValueWithBestProbability.put(probabilityValueEntry.getKey(), probabilityValueEntry.getValue());
					
				}
				// otherwise, should be added to smaller side
				else if (probabilityValueEntry.getValue() < valueWithBestDensityProbability){
					
					smallerSideOfValueWithBestProbability.put(probabilityValueEntry.getKey(), probabilityValueEntry.getValue());
				}
				
				
			}
			
			double[] boundsForAttribute = new double[2];
			


			
			if(smallerSideOfValueWithBestProbability.isEmpty()){
				boundsForAttribute[0] = valueWithBestDensityProbability;
			}else{
				boundsForAttribute[0] = smallerSideOfValueWithBestProbability.lastEntry().getValue();
			}
			
			if(greaterSideOfValueWithBestProbability.isEmpty()){
				boundsForAttribute[1] = valueWithBestDensityProbability;
			}else{
				boundsForAttribute[1] = greaterSideOfValueWithBestProbability.lastEntry().getValue();
			}
			
			return boundsForAttribute;
			
		}
		
	
	}
	
	public class Rule{
		
		// list of all relevant rule terms to build a complete rule
		List<RuleTerm> listOfRuleTerm;
		
		// rule classification which this rule refers to: condition AND condition = CLASSIFICATION
		double classification;
		
		// classification attribute
		Attribute classificationAttribute;
		
		// rule age indicated at when the rule is created (time = no. of seen instances)
		int age;
		
		// just for statistic monitor
		int instancesCoveredWhenRuleCreated;
		
		// class distribution of the rule
		double[] classDistribution;
		
		// number of instances that rule matches and correctly cover
		int noOfCovered;
		int noOfCorrectlyCovered;
		
		public Rule(int noOfClassLabels, double classificationIn, Attribute classificationAttributeIn, int ageIn){
			listOfRuleTerm = new ArrayList<>();
			classification = classificationIn;
			classificationAttribute = classificationAttributeIn;
			
			age = ageIn;
			
			classDistribution = new double[noOfClassLabels];
			
			noOfCovered++;
			noOfCorrectlyCovered++;
			
		}
		
		public void addRuleTerm(RuleTerm ruleTerm){
			listOfRuleTerm.add(ruleTerm);
		}
		
		public boolean coveredByRule(Instance instance){
			
			// check whether all rule terms are satisfied
			for (RuleTerm ruleTerm : listOfRuleTerm) {
				
				if(!ruleTerm.coveredByRuleTerm(instance)){
					return false;
				}
			}
			
			return true;
		}
		
		
		public void updateClassDistribution(Instance instance){
			
			// update class distribution of the rule with correct class label
			classDistribution[(int) instance.classValue()]++;
			
		}
		
		// return the major class in the rule class distribution
		public int classWithLargestValueInDistribution(){
			return Utils.maxIndex(classDistribution);
		}
		
		// return accuracy rate of this rule
		public double accRate(){
			return ((double)noOfCorrectlyCovered/(double)noOfCovered);
		}
		
		// return coverage with seen intances
		public double coverage(){
			return (double) noOfCovered / (double) totalSeenInstances;
		}
		
		public String printRule(){
			
			StringBuilder sb = new StringBuilder();
			int count = 0;
			// write rule term
			for (RuleTerm ruleTerm : listOfRuleTerm) {
				sb.append(ruleTerm.printRuleTerm());
				
				// add a AND operator if there are more rule term
				if(count < listOfRuleTerm.size() - 1){
					sb.append(" AND ");
				}
				
				count++;		
			}
			
			// add classification
			sb.append(" THEN ");
			sb.append(classificationAttribute.value((int)classification));
			
			sb.append(" (age:" + age + ",instsCoverdAtCreated: " + instancesCoveredWhenRuleCreated + ",acc: " + accRate() + ", cover: " + coverage() + ")");
			return sb.toString();
		}

		public void setInstancesCoveredWhenRuleCreated(
				int instancesCoveredWhenRuleCreated) {
			this.instancesCoveredWhenRuleCreated = instancesCoveredWhenRuleCreated;
		}
		
		// remove rule if accuracy drops below a threshold
		public boolean ruleShouldBeRemoved(){
			
			if(noOfCovered >= MinRuleTriesOption.getValue()){
				
				if(accRate() < ruleValidationThresholdOption.getValue()){
					return true;
				}
			}
			
			return false;
		}
	}
	
	public class RuleTerm{
		
		final static int CATEGORICAL_ATTRIBUTE = 1;
		final static int NUMERIC_ATTRIBUTE = 2;
		
		Attribute attribute;
		double value;
		
		int attribute_type;
		
		double numericUpperBound;
		double numericLowerBound;
		
		
		// constructor for categorical attribute
		public RuleTerm(Attribute attributeIn, double valueIn){
			
			attribute = attributeIn;
			
			// index of the attribute value if the attribute is categorical
			// actual value of the attribute if the attribute is numeric
			value = valueIn;	
			
			// set attribute type
			attribute_type = CATEGORICAL_ATTRIBUTE;
			
		}
		
		// constructor for numeric attribute
		public RuleTerm(Attribute attributeIn, double lowerBound, double upperBound){
			
			attribute = attributeIn;
			
			numericLowerBound = lowerBound;
			numericUpperBound = upperBound;
			
			attribute_type = NUMERIC_ATTRIBUTE;
		}
		
		public String getAttributeName(){
			return attribute.name();
		}
		
		public boolean coveredByRuleTerm(Instance instance){
			
			// check whether rule term for categorical or
			// numeric attribute
			if(attribute_type == CATEGORICAL_ATTRIBUTE){
				
				// index of the value on the attribute from instance matched value in the rule term
				if(instance.value(attribute) == value){
					return true;
				}
				
			}
			//perform for numeric attribute rule term in form: x(lower bound) < Attribute <= y(upper bound)
			else if(attribute_type == NUMERIC_ATTRIBUTE){
				
				if(instance.value(attribute) > numericLowerBound && instance.value(attribute) <= numericUpperBound){
					return true;
				}
				
			}
			
			

			
			return false;
		}
		
		public String printRuleTerm(){
			String ruleTermString = "";
			
			if(attribute_type == CATEGORICAL_ATTRIBUTE){
				ruleTermString += "[C]";
			}
			
			if(attribute_type == NUMERIC_ATTRIBUTE){
				ruleTermString += "[N]";
			}
			
			ruleTermString += attribute.name();
			
			if(attribute_type == CATEGORICAL_ATTRIBUTE){
				ruleTermString += " = ";
				ruleTermString += attribute.value((int) value);
			}
			
			if(attribute_type == NUMERIC_ATTRIBUTE){
				ruleTermString += " : ";
				
				ruleTermString += numericLowerBound;
				ruleTermString += " < (value) <= ";
				ruleTermString += numericUpperBound;
			}
			
			return ruleTermString;
		}
		
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		//ArffFileStream arffFileStream = new ArffFileStream("resources/UCI_KDD/nominal/cmc.arff", -1);
		
		// read arff file WEKA way
		DataSource source = new DataSource("data/cmc.arff");
		
		// stream generator
		RandomTreeGenerator treeGenerator = new RandomTreeGenerator();
		treeGenerator.numClassesOption.setValue(5);
		treeGenerator.numNumericsOption.setValue(0);
		treeGenerator.prepareForUse();
		
		// HoeffdingRules classifier
		GeRules gErules = new GeRules();
		gErules.prepareForUse();
		
		// load data into instances set
		Instances data = source.getDataSet();
		
		 // setting class attribute if the data format does not provide this information
		 // For example, the XRFF format saves the class attribute information as well
		 if (data.classIndex() == -1)
		   data.setClassIndex(data.numAttributes() - 1);
		 
		 // Using Prism classifier
		 //hoeffdingRules.learnRules(Collections.list(data.enumerateInstances()));
		 for (Instance instance : Collections.list(data.enumerateInstances())) {
			 gErules.trainOnInstanceImpl(instance);
                         
                         
			 gErules.correctlyClassifies(instance);
		}
                
                Instance anInstance = Collections.list(data.enumerateInstances()).get(10);
                System.out.println(anInstance);
                for (Rule aRule : gErules.RulesCoveredInstance(anInstance)) {
                    
                    System.out.println(aRule.printRule());
                }
                
                for(Rule aRule: gErules.rulesList){
                    System.out.println(aRule.printRule());
                }
                    
	}

}