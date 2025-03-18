import uuid
from pretsa import Pretsa
from anytree import PreOrderIter, find
import sys
import copy
import numpy as np
import math

class Pretsa_star(Pretsa):
    def __init__(self,eventLog,greedy=True):
        super().__init__(eventLog)
        self._queue = list()
        self._minDistanceMatrix, self.__minClosestSequenceMatrix = self._calculateMinDistances(self._distanceMatrix)
        self.__variantDictCounterName = "Counter"
        self.__variantDictTClosenessName = "t-closeness-violation"
        self.__variantDictCasesSetName = "Cases"
        self.__variantDictName = "Variant"
        self.__operationDictCaseOrigin = "cases_origin"
        self.__operationDictCutOutTraces = "cases"
        self.__operationDictCasesGoal = "cases_goal"
        self.__greedy = greedy
        self.__states = list()
        self.__closestConformingSequence = dict()
        self.__closestViolatingSequence = dict()
        self.__lastTargetSequence = None
        self.__lastStartSequence = None
        self.__caseSensitiveValues = self._extract_case_sensitive_values(eventLog)
        self.__used_nonces = set()

    def runPretsa(self,k,t,nonce):
        self._validate_nonce(nonce) # Validate the nonce to prevent replay attacks
        tree = self._tree
        i = 0
        currentCost = 0.0
        changedCases = set()
        caseToSequenceDict = self._caseToSequenceDict
        bestOption = sys.maxsize
        bestTree = None
        while True:
            if self.__greedy:
                self._queue = list()
            if self.__stateIsNew(caseToSequenceDict,changedCases):
                violatingCases, violatingVariants = self._getViolatingCases(tree, k,caseToSequenceDict)
                print(len(violatingVariants))
                if len(violatingCases) == 0 and currentCost < bestOption:
                    bestOption = currentCost
                    bestTree = tree
                    bestChangedCases = changedCases
                self._updateQueue(k,tree,violatingCases,violatingVariants, currentCost,changedCases,caseToSequenceDict)
            if not self.__shouldAlgorithmContinue(self._queue,bestOption):
                totalDistanceFromOriginalLog = bestOption
                break
            operation = self._queue.pop(0)
            tree = copy.deepcopy(operation["start"])
            tree = self._performOperation(tree,operation)
            caseToSequenceDict = self._updateCaseToSequenceDict(operation)
            currentCost = operation["realCost"]
            changedCases = operation["changedCases"]
            i += 1
        self._tree = self._addDifferentialPrivateNosieToEnsureTCloseness(bestTree,t)

        self._checkHomogenousNodes(self._tree)    # Check if the tree contains homogeneous nodes
            
        self._privacyLevel = self._checkPrivacyLevel(self._tree)    # Check if the privacy level is above the minimum threshold
        print("Current privacy level: ", self._privacyLevel)

        return bestChangedCases, totalDistanceFromOriginalLog

    def _extract_case_sensitive_values(self, eventLog): 
        case_sensitive_values = {}
        for index, row in eventLog.iterrows():
            case_id = row['Case ID']    # Chose these as sensitive attributes for bpic2013 dataset
            impact = row['impact']
            case_sensitive_values[case_id] = impact
        return case_sensitive_values
    
    def _generate_nonce(self):
        nonce = str(uuid.uuid4())
        print("Nonce generated: ", nonce)
        return nonce    # Generate a random nonce

    def _validate_nonce(self, nonce):
        if nonce in self.__used_nonces: # Check if the nonce has already been used
            raise Exception("Replay attack detected! Nonce has already been used.")
        self.__used_nonces.add(nonce)
        print("Nonce validated. No replay attack. Proceeding...")

    def _checkPrivacyLevel(self, tree):
        min_cases_per_node = float('inf')
        homogeneous_nodes = 0
        total_nodes = sum(1 for _ in PreOrderIter(tree)) - 1
    
        for node in PreOrderIter(tree):
            if node != tree: 
                # Check k-anonymity
                cases_count = len(node.cases)
                min_cases_per_node = min(min_cases_per_node, cases_count)

                # Check for homogeneous nodes (l-diversity)
                sensitive_values = set(self.__caseSensitiveValues[case] for case in node.cases)
                if len(sensitive_values) == 1:
                    homogeneous_nodes += 1
                    
        privacy_level = {
            "k_anonymity_level": min_cases_per_node,
            "homogeneous_nodes_percentage": (homogeneous_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        }

        overall_score = min_cases_per_node - (homogeneous_nodes / total_nodes)
    
        return overall_score
    
    def _checkHomogenousNodes(self, tree):   # Check if the tree contains homogeneous nodes
        print("Checking for homogeneous nodes...")
        for node in PreOrderIter(tree):
            if node != tree:
                sensitive_values = set(self.__caseSensitiveValues[case] for case in node.cases)
                if len(sensitive_values) == 1:
                    print("Homogeneous nodes found! Modifying tree...")
                    self._modify_data_to_increase_diversity(node, 4)  # Increase diversity to 4-diversity
        
    def _modify_data_to_increase_diversity(self, node, l):  # l is the minimum diversity we are trying to achieve
        sensitive_values = [self.__caseSensitiveValues[case] for case in node.cases]
        unique_values = set(sensitive_values)
        value_counts = {val: sensitive_values.count(val) for val in unique_values}
    
        print(f"Current unique values: {unique_values}, count: {len(unique_values)}, need: {l}")
    
        if len(unique_values) < l:
            added_values = set()
            current_diversity = len(unique_values) + len(added_values)
            if current_diversity < l:
                value_groups = self._group_similar_values(unique_values, l) # Group similar values for generalization   
                            
                for case in node.cases: # Apply generalization to the node's cases
                    original_value = self.__caseSensitiveValues[case]
                    for group_name, group_values in value_groups.items():
                        if original_value in group_values:
                            self.__caseSensitiveValues[case] = f"{group_name}"
                            break
            
                generalized_values = set(self.__caseSensitiveValues[case] for case in node.cases)
                if len(generalized_values) < l:
                    synthetic_values = [f"synthetic_value_{i}" for i in range(l - len(generalized_values))] # Add synthetic cases with different values
                    for i, synthetic_value in enumerate(synthetic_values):
                        synthetic_case = f"synthetic_case_{node.name}_{i}"
                        node.cases.add(synthetic_case)
                        self.__caseSensitiveValues[synthetic_case] = synthetic_value
        
            # Verify we achieved l-diversity
            final_values = set(self.__caseSensitiveValues[case] for case in node.cases)
            print(f"After modification: unique values: {final_values}, count: {len(final_values)}")
        
            if len(final_values) < l:
                print(f"Warning: Could not achieve {l}-diversity for node {node.name}. Achieved {len(final_values)}-diversity.")

        return node

    def _group_similar_values(self, values, l):
        values_list = list(values)

        num_groups = max(1, math.ceil(len(values) / l))
        group_size = max(1, math.ceil(len(values) / num_groups))
    
        groups = {}
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, len(values_list))
            if start_idx < len(values_list):
                group_values = values_list[start_idx:end_idx]
                group_name = f"Group_{i+1}"
                groups[group_name] = group_values
    
        return groups   # Dictionary with generalized values mapped to original values

    def _updateQueue(self,k,tree,violatingCases,violatingVariants,currentCost,changedCases,caseToSequenceDict):
        for variant in violatingVariants.values():
            self._addOperationsToFixVariantToQueue(variant, k, tree, violatingCases, currentCost, changedCases.copy(), caseToSequenceDict)
        if self.__greedy:
            self._queue = sorted(self._queue, key=lambda k: (k["cost"], -k["isViolating"]))
        else:
            self._queue = sorted(self._queue, key=lambda k: (k["cost"], -len(k["changedCases"])))

    def _updateCaseToSequenceDict(self,operation):
        caseToSequenceDict = operation["caseToSequenceDict"].copy()
        for case in operation[self.__operationDictCutOutTraces]:
            caseToSequenceDict[case] = operation[self.__operationDictCasesGoal]
        return caseToSequenceDict

    def _performOperation(self,tree,operation):
        node = find(tree, lambda node1: node1.sequence == operation[self.__operationDictCaseOrigin])
        self._cutCasesOutOfTreeStartingFromNode(node,operation[self.__operationDictCutOutTraces],tree)
        self.__lastTargetSequence = operation[self.__operationDictCasesGoal]
        self.__lastStartSequence = operation[self.__operationDictCaseOrigin]
        for case in operation[self.__operationDictCutOutTraces]:
            self._addCaseToTree(case, operation[self.__operationDictCasesGoal],tree)
        return tree

    def __areSequencesTheSame(self,sequence1, sequence2):
        if sequence1 == sequence2:
            return True
        else:
            return False

    def _calculateMinDistances(self,distanceMatrix):
        minDistanceMartrix = dict()
        minClosestSequenceMatrix = dict()
        for sequence in distanceMatrix.keys():
            minDistanceMartrix[sequence] = min(distanceMatrix[sequence].values())
            minClosestSequenceMatrix[sequence] = min(distanceMatrix[sequence])
        return minDistanceMartrix, minClosestSequenceMatrix

    def _getViolatingCases(self, tree, k, caseToSequenceDict):
        cases = set()
        variants = dict()
        for node in PreOrderIter(tree):
            if node != tree:
                if len(node.cases) < k:
                    newcases = set(node.cases.difference(cases))
                    cases = cases.union(node.cases)
                    for newcase in newcases:
                        variant = variants.get(caseToSequenceDict[newcase], dict())
                        variant[self.__variantDictCounterName] = variant.get(self.__variantDictCounterName, 0) + 1
                        variant[self.__variantDictCasesSetName] = variant.get(self.__variantDictCasesSetName, set())
                        variant[self.__variantDictCasesSetName].add(newcase)
                        variant[self.__variantDictName] = caseToSequenceDict[newcase]
                        variants[caseToSequenceDict[newcase]] = variant
        return cases,variants

    def _calculateDistanceHeuristic(self,k,allVariantsInTree,violatingVariants):
        distanceHeuristic = 0.0
        conformingVariants = allVariantsInTree.difference(violatingVariants.keys())
        for variant in violatingVariants:
            distanceHeuristic += min(self.__costClosestViolatingSeqeunce(k,violatingVariants.keys(),variant,violatingVariants[variant]),self.__costClosestConformingSequence(conformingVariants,variant,violatingVariants[variant]))
        return distanceHeuristic

    def __costClosestViolatingSeqeunce(self, k, violatingVariants, variantToFix, casesInVariantToFix):
        minDistance = sys.maxsize
        for variant in violatingVariants:
            if self._getDistanceSequences(variant,variantToFix) < minDistance:
                minDistance = self._getDistanceSequences(variant,variantToFix)
        result = (minDistance * min(casesInVariantToFix, abs(casesInVariantToFix - k))) / 2
        return result

    def __costClosestConformingSequence(self,conformingVariants,variantToFix,casesInVariantToFix):
        minDistance = sys.maxsize
        for variant in conformingVariants:
            if self._getDistanceSequences(variant,variantToFix) < minDistance:
                minDistance = self._getDistanceSequences(variant,variantToFix)
        result = float(minDistance * casesInVariantToFix)
        return result


    def __getViolatingVariants(self,caseToSequenceDict,violatingCases):
        violatingVariants = dict()
        for case in violatingCases:
            violatingVariants[caseToSequenceDict[case]] = violatingVariants.get(caseToSequenceDict[case],0) + 1
        return violatingVariants

    def __getRemainingViolatingVariants(self,violatingVariants,fixedCases,caseToSequenceDict):
        remainingViolatingVariants = violatingVariants.copy()
        for case in fixedCases:
            if case in remainingViolatingVariants.keys():
                del remainingViolatingVariants[caseToSequenceDict[case]]
        return remainingViolatingVariants

    def _isItNecassaryToCheckAllSequences(self,variantToFix):
        closestConformingVariant = self.__closestConformingSequence.get(variantToFix[self.__variantDictName],None)
        closestViolatingVariant = self.__closestViolatingSequence.get(variantToFix[self.__variantDictName],None)
        if not self.__greedy:
            return True
        elif self.__lastTargetSequence is None or closestConformingVariant is None or closestViolatingVariant is None:
            return True
        elif closestViolatingVariant == self.__lastTargetSequence or closestViolatingVariant == self.__lastStartSequence:
            self.__closestViolatingSequence[variantToFix[self.__variantDictName]] = None
            return True
        else:
            return False

    def _getPotentialTargetSequences(self,tree,violatingVariants,variantToFix,k):
        isSubSet = False
        if self._isItNecassaryToCheckAllSequences(variantToFix):
            variants = tree.sequences
        else:
            variants = set()
            variants.add(self.__closestViolatingSequence[variantToFix[self.__variantDictName]])
            variants.add(self.__closestConformingSequence[variantToFix[self.__variantDictName]])
            variants.add(self.__lastTargetSequence)
            isSubSet = True
        addVariants = False
        variantToRemove = set()
        for variant in variants:
            if self.__willOperationCreatesNewViolation(variantToFix[self.__variantDictName],variant,k,variantToFix[self.__variantDictCasesSetName].copy(),tree) and not self.__greedy:
                variantToRemove.add(variant)
                if variant == self.__closestViolatingSequence.get(variantToFix[self.__variantDictName],None):
                    addVariants = True
        variants.difference(variantToRemove)
        if addVariants and isSubSet:
            variants = variants.union(set(violatingVariants.keys()))
        return variants

    def _getNewBestOperationDict(self,bestOperartion, occuredCost,projectedCost,targetSequence):
        if projectedCost < bestOperartion["projectedCost"]:
            bestOperartion["occuredCost"] = occuredCost
            bestOperartion["projectedCost"] = projectedCost
            bestOperartion["targetSequence"] = targetSequence
        return bestOperartion

    def _initializeVariablesForaddOpertionsToFixVariantToQueue(self):
        bestOperationCompliant = dict()
        bestOperationCompliant["projectedCost"] = sys.maxsize
        bestOperationViolating = dict()
        bestOperationViolating["projectedCost"] = sys.maxsize
        minCostOfCurrentBestOption = sys.maxsize
        return bestOperationCompliant, bestOperationViolating,minCostOfCurrentBestOption

    def _getProjectedCost(self,violatingVariants,caseToSequenceDict,fixedCases,tree,occuredCost,k):
        remainingViolatingVariants = self.__getRemainingViolatingVariants(violatingVariants, fixedCases,caseToSequenceDict)
        distanceHeuristic = self._calculateDistanceHeuristic(k, self._getAllPotentialSequencesTree(tree),remainingViolatingVariants)
        projectedCost = distanceHeuristic + occuredCost
        return projectedCost

    def _getCasesFixedByOperation(self,variantToFix,targetNode,k):
        fixedCases = variantToFix[self.__variantDictCasesSetName].copy()
        if self.__checkIfOperationFixesTargetVariant(targetNode, fixedCases, k):
            fixedCases = fixedCases.union(targetNode.cases)
        return fixedCases

    def _addOperationWithViolatingTargetToQueue(self,bestOperationViolating,changedCases,occuredCost,projectedCost,tree,targetSequence,variantToFix,caseToSequenceDict):
        if self.__greedy:
            bestOperationViolating = self._getNewBestOperationDict(bestOperationViolating, occuredCost, projectedCost,
                                                                   targetSequence)
        else:
            self.__addOperationToQueue(projectedCost, variantToFix, tree, targetSequence, occuredCost, changedCases,
                                       caseToSequenceDict)
        return bestOperationViolating

    def _addOperationsToQueueInHeuristicPRETSA(self,variantToFix,bestOperationCompliant,bestOperationViolating,tree,changedCases,caseToSequenceDict):
        if bestOperationCompliant.get("targetSequence", None) is not None:
            self.__closestConformingSequence[variantToFix[self.__variantDictName]] = bestOperationCompliant["targetSequence"]
        if bestOperationViolating.get("targetSequence", None) is not None:
            self.__closestViolatingSequence[variantToFix[self.__variantDictName]] = bestOperationViolating[
                "targetSequence"]
            self.__addOperationToQueue(bestOperationViolating.get("projectedCost", sys.maxsize), variantToFix, tree,
                                       bestOperationViolating["targetSequence"], bestOperationViolating["occuredCost"],
                                       changedCases, caseToSequenceDict)


    def _addOperationsToFixVariantToQueue(self, variantToFix, k, tree, violatingCases, pastCost, changedCases, caseToSequenceDict):
        bestOperationCompliant, bestOperationViolating, minCostOfCurrentBestOption = self._initializeVariablesForaddOpertionsToFixVariantToQueue()
        violatingVariants = self.__getViolatingVariants(caseToSequenceDict,violatingCases)
        potentialTargetSequences = self._getPotentialTargetSequences(tree,violatingVariants,variantToFix,k)
        for targetSequence in potentialTargetSequences:
            if not self.__areSequencesTheSame(targetSequence, variantToFix[self.__variantDictName]):
                targetNode = find(tree, lambda node: node.sequence == targetSequence)
                if targetNode == None:
                    continue
                fixedCases = self._getCasesFixedByOperation(variantToFix,targetNode,k)
                costOfOperartion = self._getDistanceSequences(variantToFix[self.__variantDictName], targetSequence) * variantToFix[self.__variantDictCounterName]
                occuredCost = costOfOperartion + pastCost
                if (self.__greedy and occuredCost < minCostOfCurrentBestOption) or not self.__greedy: #If the cost by operation is higher without distance metric, there is no sense in even calculating one
                    projectedCost = self._getProjectedCost(violatingVariants,caseToSequenceDict,fixedCases,tree,occuredCost,k)
                    #Block operations that would create new violations -> otherwise the problem is not feasible
                    if len(targetNode.cases) >= k:
                        bestOperationCompliant = self._getNewBestOperationDict(bestOperationCompliant,occuredCost,projectedCost,targetSequence)
                    else:
                        bestOperationViolating = self._addOperationWithViolatingTargetToQueue(bestOperationViolating,changedCases,occuredCost,projectedCost,tree,targetSequence,variantToFix,caseToSequenceDict)
                    minCostOfCurrentBestOption = min(bestOperationViolating["projectedCost"],bestOperationCompliant["projectedCost"])
        if self.__greedy:
            self._addOperationsToQueueInHeuristicPRETSA(variantToFix,bestOperationCompliant,bestOperationViolating,tree,changedCases,caseToSequenceDict)
        if bestOperationCompliant.get("targetSequence", None) is not None:
            self.__addOperationToQueue(bestOperationCompliant["projectedCost"],variantToFix,tree,bestOperationCompliant["targetSequence"],bestOperationCompliant["occuredCost"],changedCases,caseToSequenceDict,False)

    def __checkIfOperationFixesTargetVariant(self,node,fixedCases,k):
        if node == None:
            return True
        if len(node.cases) < k and (len(node.cases) + len(fixedCases)) >= k:
            return True
        else:
            return False

    def __addOperationToQueue(self, cost, variant, tree, sequence, occuredCost, changedCases, caseToSequenceDict,isViolating=True):
        step = dict()
        step["cost"] = cost
        step["start"] = tree
        step["realCost"] = occuredCost
        step[self.__operationDictCutOutTraces] = variant[self.__variantDictCasesSetName].copy()
        step[self.__operationDictCaseOrigin] = variant[self.__variantDictName]
        step["cases_goal"] = sequence
        step["changedCases"] = changedCases.union(variant[self.__variantDictCasesSetName]).copy()
        step["caseToSequenceDict"] = caseToSequenceDict
        step["isViolating"] = int(isViolating)
        self._queue.append(step)

    def __shouldAlgorithmContinue(self,queue,bestOption):
        if len(queue) == 0:
            return False
        elif self.__greedy:
            return True
        else:
            if queue[0]["cost"] > bestOption:
                return False
            else:
                return True

    def __stateIsNew(self,currentDict,changedCases):
        if self.__greedy:
            return True
        currentState = dict()
        for changedCase in changedCases:
            currentState[changedCase] = currentDict[changedCase]
        for state in self.__states:
            if self.__stateAreEqual(state,currentState):
                return False
        self.__states.append(currentState)
        return True

    def __stateAreEqual(self,state1,state2):
        if len(state1) != len(state2):
            return False
        if len(set(state1.keys()).difference(set(state2.keys()))) > 0:
            return False
        for key in state1.keys():
            if key in state2:
                if state1[key] != state2[key]:
                    return False
            else:
                return False
        return True

    def __willOperationCreatesNewViolation(self,variantToFix,targetVariant,k,casesInVariantToFix,tree):
        nodeVariantToFix = tree
        nodeTargetVariant = tree
        activitiesTargetVariant = targetVariant.split("@")
        activitiesVariantToFix = variantToFix.split("@")
        while not nodeVariantToFix is None:
            if not nodeVariantToFix == nodeTargetVariant:
                if len(nodeVariantToFix.cases) >= k:
                    if len(nodeVariantToFix.cases) < len(casesInVariantToFix) + k and not len(nodeVariantToFix.cases) == casesInVariantToFix:
                        return True
            if len(nodeTargetVariant.children) != 0:
                nodeTargetVariant = self.__getChildNodeForCertainActivity(activitiesTargetVariant.pop(),nodeTargetVariant)
            nodeVariantToFix = self.__getChildNodeForCertainActivity(activitiesVariantToFix.pop(), nodeVariantToFix)
            if nodeTargetVariant is None:
                return True
        return False

    def __getChildNodeForCertainActivity(self,nextActivity,node):
        for child in node.children:
            if child.name == nextActivity:
                return child
        return None

    def _addDifferentialPrivateNosieToEnsureTCloseness(self,tree, t):
        activityCountMap = self._retrieveNumberOfEventsPerActivity(tree)
        for node in PreOrderIter(tree):
            if node != tree:
                for annotationKey in node.annotations.keys():
                    numberOfCasesInNode = len(node.cases)
                    numberOfCasesInDistribution = activityCountMap[node.name]
                    numerator = (((t*numberOfCasesInDistribution)/numberOfCasesInNode)-1) * numberOfCasesInNode
                    denominator = numberOfCasesInDistribution - numberOfCasesInNode - 1
                    if numerator <= 0 or denominator <= 0:
                        raise Exception("t is too low. Please modify the parameter.")
                    if numberOfCasesInNode != numberOfCasesInDistribution:
                        epsilon = math.log(numerator/denominator)
                        node.annotations[annotationKey] = node.annotations[annotationKey] + np.random.laplace(scale=epsilon)
        return tree

    def _retrieveNumberOfEventsPerActivity(self,tree):
        activityCountMap = dict()
        for node in PreOrderIter(tree):
            if node != tree:
                activityCountMap[node.name] = activityCountMap.get(node.name,0) + len(node.cases)
        return activityCountMap