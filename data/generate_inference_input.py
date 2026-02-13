import sys
import json

def get_valid_input(prompt_text):
    """
    Helper to ensure user enters a float between 1.0 and 7.0.
    """
    while True:
        try:
            val_str = input(f"\n{prompt_text}\n(1-7): ")
            val = float(val_str)
            if 1.0 <= val <= 7.0:
                return val
            else:
                print(">>> Error: Please enter a number between 1.0 and 7.0.")
        except ValueError:
            print(">>> Error: Invalid input. Please enter a number.")

def main():
    print("========================================================")
    print("   AI CAREER PATH SYSTEM - INFERENCE INPUT GENERATOR    ")
    print("========================================================")
    print("Instructions: Answer each question on a scale of 1 to 7.")
    print("1 = Strongly Disagree | 4 = Neutral | 7 = Strongly Agree")
    print("========================================================\n")

    # 1. The Exact Input Order Required by Your Model
    # (Job_Label excluded)
    FEATURE_ORDER = [
        'Artistic', 'Conventional', 'Enterprising', 'Investigative', 'Realistic', 'Social',
        'Achievement Orientation', 'Adaptability', 'Attention to Detail', 'Cautiousness',
        'Cooperation', 'Dependability', 'Empathy', 'Humility', 'Initiative',
        'Innovation', 'Integrity', 'Intellectual Curiosity', 'Leadership Orientation',
        'Optimism', 'Perseverance', 'Self-Confidence', 'Self-Control', 'Sincerity',
        'Social Orientation', 'Stress Tolerance', 'Tolerance for Ambiguity'
    ]

    # 2. RIASEC Questions (5 per type)
    # The key must match the feature name in FEATURE_ORDER
    riasec_questions = {
        'Realistic': [
            "1. I would like to build kitchen cabinets.",
            "2. I would like to lay brick or tile.",
            "3. I would like to repair household appliances.",
            "4. I would like to raise fish in a fish hatchery.",
            "5. I would like to assemble electronic parts."
        ],
        'Investigative': [
            "1. I would like to study ways to reduce water pollution.",
            "2. I would like to conduct chemical experiments.",
            "3. I would like to study the movement of planets.",
            "4. I would like to examine blood samples using a microscope.",
            "5. I would like to investigate the cause of a fire."
        ],
        'Artistic': [
            "1. I would like to write books or plays.",
            "2. I would like to play a musical instrument.",
            "3. I would like to compose or arrange music.",
            "4. I would like to draw pictures.",
            "5. I would like to create special effects for movies."
        ],
        'Social': [
            "1. I would like to teach an individual an exercise routine.",
            "2. I would like to help people with personal or emotional problems.",
            "3. I would like to give career guidance to people.",
            "4. I would like to perform rehabilitation therapy.",
            "5. I would like to do volunteer work at a non-profit organization."
        ],
        'Enterprising': [
            "1. I would like to buy and sell stocks and bonds.",
            "2. I would like to manage a retail store.",
            "3. I would like to sell telephone cables to major corporations.",
            "4. I would like to negotiate business contracts.",
            "5. I would like to represent a client in a lawsuit."
        ],
        'Conventional': [
            "1. I would like to develop a spreadsheet using computer software.",
            "2. I would like to proofread records or forms.",
            "3. I would like to load computer software into a large computer network.",
            "4. I would like to operate a calculator.",
            "5. I would like to keep shipping and receiving records."
        ]
    }

    # 3. Work Style Questions (1 per type)
    work_style_questions = {
        'Achievement Orientation': "I set personally challenging goals and exert a high level of effort to master tasks and succeed.",
        'Adaptability': "I am flexible and good at adapting to new environments, changing plans, or shifting priorities on the fly.",
        'Attention to Detail': "I am careful about small details and prefer to be thorough and precise in completing work tasks.",
        'Cautiousness': "I tend to make decisions carefully, preferring to think through all consequences before taking action.",
        'Cooperation': "I am pleasant with others on the job and display a good-natured, cooperative attitude.",
        'Dependability': "I am reliable, responsible, and dependable in fulfilling obligations and meeting deadlines.",
        'Empathy': "I am sensitive to the needs and feelings of others and am capable of understanding their perspective.",
        'Humility': "I do not seek the spotlight for myself and am willing to admit when I am wrong or need help.",
        'Initiative': "I am willing to take on new responsibilities and challenges proactively without being told to do so.",
        'Innovation': "I use creativity and alternative thinking to develop new ideas or novel solutions to work-related problems.",
        'Integrity': "I value honesty and ethical behavior above all else, acting with strong moral principles.",
        'Intellectual Curiosity': "I have a strong desire to learn new things and enjoy exploring complex ideas, theories, or concepts.",
        'Leadership Orientation': "I am willing to take charge, offer opinions, and provide direction and guidance to others.",
        'Optimism': "I usually expect the best to happen and maintain a positive, hopeful outlook even during difficulties.",
        'Perseverance': "I persist in the face of obstacles and do not give up easily, even when tasks become difficult.",
        'Self-Confidence': "I feel good about my own abilities and am confident in my judgment and decisions.",
        'Self-Control': "I maintain composure and keep my emotions in check, avoiding aggressive behavior even in difficult situations.",
        'Sincerity': "I act genuinely and transparently, without pretending to be someone I am not.",
        'Social Orientation': "I prefer to work with others rather than alone and enjoy being personally connected with my coworkers.",
        'Stress Tolerance': "I accept criticism well and deal calmly and effectively with high-stress situations.",
        'Tolerance for Ambiguity': "I function well in situations where the rules are not clear, the structure is loose, or the future is uncertain."
    }

    final_input_vector = []

    # 4. Main Logic Loop
    # Iterate specifically through FEATURE_ORDER to ensure the output list is correct.
    
    for feature in FEATURE_ORDER:
        print(f"\n--- Section: {feature} ---")
        
        # Check if it's a RIASEC feature (needs averaging 5 questions)
        if feature in riasec_questions:
            questions = riasec_questions[feature]
            total_score = 0
            for q in questions:
                score = get_valid_input(q)
                total_score += score
            
            average_score = total_score / 5.0
            print(f"   -> Calculated {feature} Score: {average_score:.2f}")
            final_input_vector.append(average_score)
            
        # Check if it's a Work Style feature (single question)
        elif feature in work_style_questions:
            q = work_style_questions[feature]
            score = get_valid_input(q)
            final_input_vector.append(score)
            
        else:
            print(f"CRITICAL ERROR: Feature '{feature}' not found in question banks.")
            return

    # 5. Output Results
    print("\n========================================================")
    print("                 GENERATION COMPLETE                    ")
    print("========================================================")
    print("\nCopy this list directly into your inference script:")
    print(final_input_vector)
    
    # Optional: Save to JSON file
    with open("inference_input.json", "w") as f:
        json.dump(final_input_vector, f)
    print("\n(Also saved to 'inference_input.json')")

if __name__ == "__main__":
    main()