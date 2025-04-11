import math
import random
import time
from datetime import datetime

class CyberBullyingDetector:
    def __init__(self):
        self.vocabulary = set()
        self.word_counts_bullying = {}
        self.word_counts_normal = {}
        self.bullying_messages_count = 0
        self.normal_messages_count = 0
        self.is_trained = False
        self.feature_weights = {}
        self.confidence_threshold = 0.65
        self.context_history = []
        self.last_update = "2025-04-09"
        self.version = "3.0.1"
        
        # Pre-defined training data
        self.training_data = []
        
        # High-risk word categories (severity-weighted)
        self.high_risk_words = {
            'suicide': ["kill yourself", "kys", "end it all", "suicide", "just die", "hang yourself", "slit your", "better off dead"],
            'threats': ["beat you up", "hurt you", "kill you", "find you", "hunt you down", "come after you"],
            'discrimination': ["retard", "faggot", "gay", "nigger", "spic", "bitch", "slut", "whore", "cunt"],
            'insults': ["worthless", "pathetic", "ugly", "stupid", "fat", "loser", "freak", "failure"]
        }
        
        # Context modifiers - words that change the meaning of surrounding text
        self.context_modifiers = {
            "intensifiers": ["very", "really", "extremely", "totally", "absolutely", "completely", "so"],
            "negators": ["not", "don't", "doesn't", "isn't", "aren't", "never", "no"],
            "conditionals": ["if", "would", "could", "might", "maybe", "perhaps"],
            "sarcasm_indicators": ["sure", "right", "yeah", "whatever", "obviously"]
        }
        
        # Initialize feature weights (will be updated during training)
        self._initialize_feature_weights()

    def _initialize_feature_weights(self):
        """Initialize feature weights based on domain knowledge"""
        # Default weight for all features
        self.feature_weights = {"_default": 1.0}
        
        # Higher weights for high-risk categories
        for category, terms in self.high_risk_words.items():
            for term in terms:
                if category == 'suicide':
                    self.feature_weights[term] = 5.0
                elif category == 'threats':
                    self.feature_weights[term] = 3.5
                elif category == 'discrimination':
                    self.feature_weights[term] = 4.0
                elif category == 'insults':
                    self.feature_weights[term] = 2.5
        
        # Special pattern features
        self.feature_weights["MULTIPLE_EXCLAMATIONS"] = 1.5
        self.feature_weights["HIGH_UPPERCASE"] = 1.8
        self.feature_weights["REPEATED_MESSAGES"] = 2.0
        self.feature_weights["IMPERATIVE_TONE"] = 1.3

    def preprocess(self, text):
        """Convert text to lowercase and split into words with advanced preprocessing"""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize common text substitutions
        text = text.replace("u r", "you are")
        text = text.replace("ur", "your")
        text = text.replace("2", "to")
        text = text.replace("4", "for")
        
        # Handle emoji representations
        text = text.replace(":)", "smile")
        text = text.replace(":(", "sad")
        text = text.replace(":D", "laugh")
        
        # Tokenize (split into words)
        words = text.split()
        
        # Clean tokens
        cleaned_words = []
        for word in words:
            # Remove non-alphanumeric characters
            clean_word = ''.join([c for c in word if c.isalnum() or c == '_' or c == '-'])
            if clean_word:  # Only add non-empty words
                cleaned_words.append(clean_word)
        
        return cleaned_words

    def extract_features(self, text):
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic preprocessing
        words = self.preprocess(text)
        text_lower = text.lower()
        
        # Single word features (unigrams)
        for word in words:
            features[word] = features.get(word, 0) + 1
        
        # Word pair features (bigrams)
        if len(words) > 1:
            for i in range(len(words) - 1):
                bigram = words[i] + "_" + words[i + 1]
                features[bigram] = features.get(bigram, 0) + 1
        
        # Word triplet features (trigrams) for better context
        if len(words) > 2:
            for i in range(len(words) - 2):
                trigram = words[i] + "_" + words[i + 1] + "_" + words[i + 2]
                features[trigram] = features.get(trigram, 0) + 1
        
        # High-risk word category features
        for category, terms in self.high_risk_words.items():
            category_present = False
            for term in terms:
                if term in text_lower:
                    category_present = True
                    features[f"CATEGORY_{category.upper()}"] = 1
                    break
            if category_present:
                features[f"CATEGORY_{category.upper()}"] = 1
        
        # Context modifier features
        for modifier_type, modifiers in self.context_modifiers.items():
            for modifier in modifiers:
                if modifier in words:
                    features[f"MODIFIER_{modifier_type.upper()}"] = 1
                    
                    # Check for nearby high-risk words (within 3 words)
                    modifier_index = words.index(modifier)
                    start_idx = max(0, modifier_index - 3)
                    end_idx = min(len(words), modifier_index + 4)
                    context_window = ' '.join(words[start_idx:end_idx])
                    
                    # Check if any high-risk word is in this context window
                    for category, terms in self.high_risk_words.items():
                        for term in terms:
                            if term in context_window:
                                # Create a contextual feature
                                features[f"CONTEXT_{modifier_type.upper()}_{category.upper()}"] = 1
        
        # Statistical features
        exclamation_count = text.count('!')
        question_count = text.count('?')
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Add these statistical features
        features["HAS_EXCLAMATION"] = 1 if exclamation_count > 0 else 0
        features["MULTIPLE_EXCLAMATIONS"] = 1 if exclamation_count > 1 else 0
        features["HAS_QUESTION"] = 1 if question_count > 0 else 0
        features["HIGH_UPPERCASE"] = 1 if uppercase_ratio > 0.3 else 0
        features["SHORT_MESSAGE"] = 1 if word_count < 5 else 0
        features["LONG_MESSAGE"] = 1 if word_count > 15 else 0
        features["AVG_WORD_LENGTH"] = avg_word_length
        
        # Advanced pattern detection
        # Check for imperative sentences (commands)
        if words and words[0] in ["go", "stop", "shut", "kill", "get", "leave", "die"]:
            features["IMPERATIVE_TONE"] = 1
        
        # Check for repeated words (stuttering effect often used in bullying)
        for i in range(len(words) - 1):
            if words[i] == words[i+1] and len(words[i]) > 2:  # Ignore small words like "ha ha"
                features["REPEATED_WORDS"] = features.get("REPEATED_WORDS", 0) + 1
        
        # Check for repeated punctuation (!!!, ???)
        if '!!' in text or '???' in text:
            features["REPEATED_PUNCTUATION"] = 1
        
        # Message similarity to previous messages (for repeated harassment)
        if self.context_history:
            max_similarity = max(self._calculate_similarity(text, prev) for prev in self.context_history)
            if max_similarity > 0.7:  # High similarity threshold
                features["REPEATED_MESSAGES"] = 1
        
        return features

    def _calculate_similarity(self, text1, text2):
        """Calculate simple Jaccard similarity between two texts"""
        words1 = set(self.preprocess(text1))
        words2 = set(self.preprocess(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def train(self, update_weights=True):
        """Train the model with feature weighting"""
        # Reset counters
        self.vocabulary = set()
        self.word_counts_bullying = {}
        self.word_counts_normal = {}
        self.bullying_messages_count = 0
        self.normal_messages_count = 0
        
        # Track feature effectiveness for weight adjustment
        feature_effectiveness = {}
        
        # Process each training example
        for text, label in self.training_data:
            features = self.extract_features(text)
            
            # Update vocabulary
            self.vocabulary.update(features.keys())
            
            # Track which features appear in which class
            for feature in features:
                if feature not in feature_effectiveness:
                    feature_effectiveness[feature] = {"bullying": 0, "normal": 0, "total": 0}
                
                if label == 1:  # Bullying
                    feature_effectiveness[feature]["bullying"] += 1
                else:  # Normal
                    feature_effectiveness[feature]["normal"] += 1
                
                feature_effectiveness[feature]["total"] += 1
            
            # Update appropriate counters based on label
            if label == 1:  # Bullying
                self.bullying_messages_count += 1
                for word, count in features.items():
                    self.word_counts_bullying[word] = self.word_counts_bullying.get(word, 0) + count
            else:  # Normal
                self.normal_messages_count += 1
                for word, count in features.items():
                    self.word_counts_normal[word] = self.word_counts_normal.get(word, 0) + count
        
        # Update feature weights based on their predictive power
        if update_weights:
            self._update_feature_weights(feature_effectiveness)
        
        self.is_trained = True
        return True

    def _update_feature_weights(self, feature_effectiveness):
        """Update feature weights based on their predictive power"""
        for feature, stats in feature_effectiveness.items():
            if stats["total"] < 3:  # Skip features with too few occurrences
                continue
            
            # Calculate how strongly this feature predicts either class
            if stats["bullying"] > 0 and stats["normal"] > 0:
                bullying_ratio = stats["bullying"] / stats["total"]
                
                # Features that strongly indicate bullying get higher weights
                if bullying_ratio > 0.8:
                    self.feature_weights[feature] = 3.0
                elif bullying_ratio > 0.6:
                    self.feature_weights[feature] = 2.0
                elif bullying_ratio < 0.2:  # Strongly indicates normal content
                    self.feature_weights[feature] = 0.5
            elif stats["bullying"] > 0:  # Only appears in bullying
                self.feature_weights[feature] = 2.5
            elif stats["normal"] > 0:  # Only appears in normal
                self.feature_weights[feature] = 0.5

    def predict_probability(self, text):
        """Predict the probability that a message is bullying using weighted features"""
        if not self.is_trained:
            self.train()
        
        # Update context history
        if len(self.context_history) >= 5:
            self.context_history.pop(0)
        self.context_history.append(text)
        
        # Extract features from the text
        features = self.extract_features(text)
        
        # Calculate prior probabilities
        p_bullying = self.bullying_messages_count / (self.bullying_messages_count + self.normal_messages_count)
        p_normal = self.normal_messages_count / (self.bullying_messages_count + self.normal_messages_count)
        
        # Initialize log probabilities
        log_prob_bullying = math.log(p_bullying)
        log_prob_normal = math.log(p_normal)
        
        # Total word counts (for Laplace smoothing)
        total_bullying_words = sum(self.word_counts_bullying.values())
        total_normal_words = sum(self.word_counts_normal.values())
        vocabulary_size = len(self.vocabulary)
        
        # Calculate conditional probabilities using Laplace smoothing and feature weights
        for feature, count in features.items():
            # Get feature weight (default to 1.0 if not explicitly set)
            weight = self.feature_weights.get(feature, self.feature_weights.get("_default", 1.0))
            
            # Probability of feature given bullying
            feature_count_bullying = self.word_counts_bullying.get(feature, 0)
            p_feature_given_bullying = (feature_count_bullying + 1) / (total_bullying_words + vocabulary_size)
            log_prob_bullying += count * weight * math.log(p_feature_given_bullying)
            
            # Probability of feature given normal
            feature_count_normal = self.word_counts_normal.get(feature, 0)
            p_feature_given_normal = (feature_count_normal + 1) / (total_normal_words + vocabulary_size)
            log_prob_normal += count * weight * math.log(p_feature_given_normal)
        
        # Convert log probabilities to actual probabilities
        prob_bullying = math.exp(log_prob_bullying)
        prob_normal = math.exp(log_prob_normal)
        
        # Normalize to get final probability
        bullying_percentage = (prob_bullying / (prob_bullying + prob_normal)) * 100
        
        # Apply contextual heuristics
        bullying_percentage = self.apply_contextual_analysis(text, features, bullying_percentage)
        
        return min(max(0, bullying_percentage), 100)  # Clamp between 0-100
    
    def apply_contextual_analysis(self, text, features, base_percentage):
        """Apply advanced contextual analysis to refine prediction"""
        text_lower = text.lower()
        words = self.preprocess(text)
        
        # Apply high-risk word category boosts
        for category, terms in self.high_risk_words.items():
            for term in terms:
                if term in text_lower:
                    # Different categories get different boost levels
                    if category == 'suicide':
                        base_percentage = min(base_percentage * 1.5, 100)
                    elif category == 'threats':
                        base_percentage = min(base_percentage * 1.35, 100)
                    elif category == 'discrimination':
                        base_percentage = min(base_percentage * 1.4, 100)
                    break  # Only apply boost once per category
        
        # Check for negation patterns that may reverse meaning
        negation_terms = ["not", "never", "no", "don't", "doesn't", "didn't", "isn't", "aren't"]
        positive_terms = ["good", "great", "nice", "kind", "smart", "cool", "awesome"]
        
        # Look for negation + positive term patterns which might indicate bullying
        for i, word in enumerate(words):
            if word in negation_terms and i < len(words) - 1:
                # Check if next few words contain a positive term
                for j in range(1, min(4, len(words) - i)):
                    if words[i + j] in positive_terms:
                        base_percentage = min(base_percentage + 10, 100)  # Modest boost
                        break
        
        # Check for bullying indicators based on message history context
        if len(self.context_history) > 1:
            # If multiple similar messages, increase likelihood (harassment pattern)
            similar_message_count = sum(1 for m in self.context_history[:-1] 
                                      if self._calculate_similarity(m, text) > 0.6)
            if similar_message_count > 0:
                base_percentage = min(base_percentage + (similar_message_count * 5), 100)
        
        # Message length considerations
        word_count = len(words)
        if word_count < 5:  # Very short messages
            if base_percentage > 50:  # If already leaning toward bullying
                base_percentage += 5  # Short aggressive messages are more impactful
        elif word_count > 20:  # Very long messages
            if base_percentage > 60:  # If strongly leaning toward bullying
                base_percentage -= 5  # Dilute slightly for very long messages
        
        # Special feature considerations
        if "IMPERATIVE_TONE" in features and base_percentage > 30:
            base_percentage += 5
            
        if "REPEATED_WORDS" in features and features["REPEATED_WORDS"] > 1:
            base_percentage += features["REPEATED_WORDS"] * 2
            
        if "HIGH_UPPERCASE" in features and "MULTIPLE_EXCLAMATIONS" in features:
            base_percentage += 10  # Shouting + exclamation is aggressive
        
        # Calculate confidence based on feature presence
        confidence = self._calculate_prediction_confidence(features)
        
        # Apply confidence adjustment
        if confidence < self.confidence_threshold:
            # Move toward 50% for low confidence predictions
            adjustment = (0.5 - confidence) * 0.5
            base_percentage = base_percentage * (1 - adjustment) + 50 * adjustment
        
        return base_percentage

    def _calculate_prediction_confidence(self, features):
        """Calculate confidence level in prediction based on feature diagnosticity"""
        # Start with base confidence
        confidence = 0.5
        
        # Count strong predictive features present
        strong_features = 0
        weak_features = 0
        
        for feature in features:
            weight = self.feature_weights.get(feature, 1.0)
            if weight > 1.8:  # Strong predictive feature
                strong_features += 1
            elif weight < 0.7:  # Weak or contradictory feature
                weak_features += 1
        
        # Adjust confidence based on features
        confidence += min(0.4, strong_features * 0.1)  # Max +0.4 from strong features
        confidence -= min(0.3, weak_features * 0.05)   # Max -0.3 from weak features
        
        # Ensure confidence is between 0 and 1
        return max(0.1, min(0.95, confidence))

    def add_training_example(self, text, is_bullying):
        """Add a new training example and update the model"""
        label = 1 if is_bullying else 0
        self.training_data.append((text, label))
        
        # Incrementally update the model
        features = self.extract_features(text)
        
        # Update vocabulary
        self.vocabulary.update(features.keys())
        
        # Update appropriate counters based on label
        if label == 1:  # Bullying
            self.bullying_messages_count += 1
            for word, count in features.items():
                self.word_counts_bullying[word] = self.word_counts_bullying.get(word, 0) + count
        else:  # Normal
            self.normal_messages_count += 1
            for word, count in features.items():
                self.word_counts_normal[word] = self.word_counts_normal.get(word, 0) + count
        
        # Every 10 new examples, update feature weights
        if (len(self.training_data) % 10) == 0:
            self.train(update_weights=True)
        
        self.is_trained = True
        self.last_update = datetime.now().strftime("%Y-%m-%d")
        return True
    
    def get_feature_importance(self, text):
        """Analyze which features contributed most to the classification"""
        if not self.is_trained:
            self.train()
            
        features = self.extract_features(text)
        feature_contributions = []
        
        # Calculate contribution of each feature
        for feature, count in features.items():
            # Get feature weight
            weight = self.feature_weights.get(feature, self.feature_weights.get("_default", 1.0))
            
            # Calculate how much this feature appears in each class
            feature_count_bullying = self.word_counts_bullying.get(feature, 0)
            feature_count_normal = self.word_counts_normal.get(feature, 0)
            
            # Calculate total occurrences
            total_bullying = max(1, self.bullying_messages_count)
            total_normal = max(1, self.normal_messages_count)
            
            # Calculate ratio of appearance in bullying vs normal
            bullying_ratio = feature_count_bullying / total_bullying
            normal_ratio = feature_count_normal / total_normal
            
            # Skip features that don't differentiate well
            if feature_count_bullying == 0 and feature_count_normal == 0:
                continue
                
            # Calculate the feature's contribution to bullying classification
            if bullying_ratio > normal_ratio:
                contribution = weight * (bullying_ratio / (normal_ratio + 0.01))
                direction = "bullying"
            else:
                contribution = weight * (normal_ratio / (bullying_ratio + 0.01))
                direction = "normal"
                
            feature_contributions.append((feature, contribution, direction))
        
        # Sort by contribution
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top contributions
        return feature_contributions[:10]
    
    def get_explanation(self, text, prediction):
        """Generate a human-readable explanation for the prediction"""
        top_features = self.get_feature_importance(text)
        
        if prediction < 30:
            explanation = "This message appears to be non-bullying because:\n"
        elif prediction > 70:
            explanation = "This message was identified as likely bullying because:\n"
        else:
            explanation = "This message has mixed indicators:\n"
        
        # Add top features to explanation
        for feature, contribution, direction in top_features:
            if feature.startswith("CATEGORY_"):
                category = feature.replace("CATEGORY_", "").lower()
                explanation += f"- It contains words from the {category} category\n"
            elif feature.startswith("MODIFIER_"):
                modifier_type = feature.replace("MODIFIER_", "").lower()
                explanation += f"- It uses {modifier_type} language\n"
            elif feature.startswith("CONTEXT_"):
                context_parts = feature.replace("CONTEXT_", "").lower().split("_")
                explanation += f"- It combines {context_parts[0]} with {context_parts[1]} content\n"
            elif feature in ["HIGH_UPPERCASE", "MULTIPLE_EXCLAMATIONS", "REPEATED_PUNCTUATION"]:
                explanation += f"- It uses aggressive formatting (caps, punctuation)\n"
            elif "_" in feature and not feature.isupper():
                # It's likely a bigram or trigram
                words = feature.replace("_", " ")
                explanation += f"- It contains the phrase '{words}'\n"
            elif contribution > 2.0:
                explanation += f"- It uses the word '{feature}'\n"
        
        return explanation

    def get_model_statistics(self):
        """Return statistics about the model's current state"""
        return {
            "version": self.version,
            "last_updated": self.last_update,
            "training_examples": len(self.training_data),
            "bullying_examples": self.bullying_messages_count,
            "normal_examples": self.normal_messages_count,
            "vocabulary_size": len(self.vocabulary),
            "feature_count": len(self.feature_weights),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def save_to_file(self, filename):
        """Save model data to a Python file that can be imported later"""
        data = {
            "version": self.version,
            "bullying_count": self.bullying_messages_count,
            "normal_count": self.normal_messages_count,
            "bullying_words": self.word_counts_bullying,
            "normal_words": self.word_counts_normal,
            "feature_weights": self.feature_weights,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            f.write(f"# Bullying Detection Model - Generated on {data['timestamp']}\n\n")
            f.write("model_data = {\n")
            for key, value in data.items():
                f.write(f"    '{key}': {repr(value)},\n")
            f.write("}\n")
        
        return True


def main():
    print("Advanced Bullying Detection System with Machine Learning")
    print("======================================================")
    print("Type a sentence to analyze or type a command:")
    print("- 'quit': Exit the program")
    print("- 'stats': View model statistics")
    print("- 'save': Save the current model to a file")
    print(f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Training model with initial dataset...")
    
    # Initialize and train the model
    detector = CyberBullyingDetector()
    detector.train()
    print("Initial training complete!")
    
    while True:
        user_input = input("\nEnter a sentence or command: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            stats = detector.get_model_statistics()
            print("\nModel Statistics:")
            for key, value in stats.items():
                print(f"- {key}: {value}")
            continue
        elif user_input.lower() == 'save':
            filename = f"bullying_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            detector.save_to_file(filename)
            print(f"Model saved to {filename}")
            continue
        
        start_time = time.time()
        result = detector.predict_probability(user_input)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Display results with confidence information
        confidence = detector._calculate_prediction_confidence(detector.extract_features(user_input))
        confidence_str = f"Confidence: {confidence*100:.1f}%"
        
        print(f"\nBullying likelihood: {result:.2f}% ({confidence_str})")
        print(f"Processing time: {processing_time:.2f} ms")
        
        # Provide interpretation
        if result < 20:
            print("Interpretation: Very low likelihood of bullying content.")
        elif result < 40:
            print("Interpretation: Some concerning elements, but likely not bullying.")
        elif result < 60:
            print("Interpretation: Moderate concern - contains potentially harmful content.")
        elif result < 80:
            print("Interpretation: High concern - likely contains bullying content.")
        else:
            print("Interpretation: Very high concern - highly likely to be bullying content.")
        
        # Show explanation of the prediction
        print("\nExplanation:")
        explanation = detector.get_explanation(user_input, result)
        print(explanation)
        
        # Optional feedback loop to improve the model
        if result > 15 and result < 90:  # Only ask for feedback on non-trivial predictions
            feedback = input("\nIs this prediction accurate? (y/n, or press Enter to skip): ")
            if feedback.lower() == 'n':
                correct_label = input("Is this message bullying? (y/n): ")
                is_bullying = correct_label.lower() == 'y'
                detector.add_training_example(user_input, is_bullying)
                print("Thank you for the feedback! Model has been updated.")


if __name__ == "__main__":
    main()