import logging
import ollama

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
reasoning_modules = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternative perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "16. What is the core issue or problem that needs to be addressed?",
    "17. What are the underlying causes or factors contributing to the problem?",
    "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "19. What are the potential obstacles or challenges that might arise in solving this problem?",
    "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "23. How can progress or success in solving the problem be measured or evaluated?",
    "24. What indicators or metrics can be used?",
    "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "30. Is the problem a design challenge that requires creative solutions and innovation?",
    "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "33. What kinds of solution typically are produced for this kind of problem specification?",
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    "38. Let’s think step by step."
    "39. Let’s make a step by step plan and implement it with good notation and explanation.",
]


def respond(query, messages):
    logging.info(f'Received query: {query}')
    messages.append({"role": "user", "content": query})
    response = ollama.chat(model='gemma:2b-instruct', messages=messages)
    messages.append({"role": "assistant", "content": response['message']['content']})
    r = response['message']['content']
    logging.info(f'Response: {r}')
    return r, messages

def init_prompt(modules, objective):
    prompt = f"""
    Select several reasoning modules that are crucial to utilize in order to solve the given task:

    All reasoning module descriptions:
    {modules}

    Task: {objective}

    Select several modules are crucial for solving the task above:

    """
    logging.info('Initialized prompt for module selection')
    return prompt

def structured_prompt(modules, objective):
    prompt = f"""
    Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format:

    Here's an example:

    Example task:

    If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

    Example reasoning structure:

    Adapted module description:
    {modules}

    Task: {objective}

    Implement a reasoning structure for solvers to follow step-by-step and arrive at correct answer.

    Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can fill it out and arrive at the correct conclusion for tasks like this
    """
    logging.info('Initialized structured prompt for task operationalization')
    return prompt

def reasoning_prompt(structure, description):
    prompt = f"""
    Follow the step-by-step reasoning plan in JSON to correctly solve the task. Fill in the values following the keys by reasoning specifically about the task given. Do not simply rephrase the keys.
        
    Reasoning Structure:
    {structure}

    Task: {description}
    
    DO NOT FORGET TO GIVE FINAL ANSWER in the following format:
    **Conclusion:**
    """
    logging.info('Initialized reasoning prompt for solving the task')
    return prompt

import re

def extract_conclusion(text):
    # Pattern to match the conclusion section
    conclusion_pattern = re.compile(r"\*\*Conclusion:\*\*(.*?)$", re.DOTALL)
    match = conclusion_pattern.search(text)
    if match:
        # Extract and return the conclusion text
        return match.group(1).strip()
    else:
        logging.ERROR(ValueError("Conclusion not found in text"))
        return text

def main():
    logging.info('Starting main function')
    messages = [{"role":"system", "content":" You are a helpful assistant."}]
    task_example = "Lisa has 10 apples. She gives 3 apples to her friend and then buys 5 more apples from the store. How many apples does Lisa have now?"
    reasoning_modules_str = "\n".join(reasoning_modules)
    selected_modules, messages = respond(init_prompt(reasoning_modules_str, task_example), messages)
    logging.info(f'Selected modules: {selected_modules}')
    structure, messages = respond(structured_prompt(selected_modules, task_example), messages)
    logging.info(f'Structured prompt: {structure}')
    final_answer, messages = respond(reasoning_prompt(structure, task_example), messages)
    logging.info(f'Final answer: {final_answer}')
    conclusion = extract_conclusion(final_answer)
    logging.info(f'Conclusion: {conclusion}')
    

if __name__ == "__main__":
    main()

