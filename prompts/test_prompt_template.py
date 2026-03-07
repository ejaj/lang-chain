import sys 
from prompt_template import (
    format_prompt,
    format_prompt_safe,
    format_prompt_validated,
    PromptTemplate,
    ConversationTemplate,
    FewShotPrompt,
    ChainOfThoughtPrompt
)

def test_format_prompt():
    """Test basic format_prompt function."""
    print("\n" + "="*60)
    print("TEST 1: format_prompt()")
    print("="*60)
    # Test 1.1: Basic formatting
    template = "Hello {name}, you are {age} years old"
    result = format_prompt(template, name="Alice", age=25)
    expected = "Hello Alice, you are 25 years old"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("Test 1.1 passed: Basic formatting")
    
    # Test 1.2: Single variable
    result = format_prompt("Welcome {user}!", user="Bob")
    expected = "Welcome Bob!"
    assert result == expected
    print("Test 1.2 passed: Single variable")
    # Test 1.3: No variables
    result = format_prompt("Hello World")
    expected = "Hello World"
    assert result == expected
    print("Test 1.3 passed: No variables")
    
    # Test 1.4: Multiple same variables
    result = format_prompt("{name} loves {name}", name="Charlie")
    expected = "Charlie loves Charlie"
    assert result == expected
    print("Test 1.4 passed: Duplicate variables")
    
    print("All format_prompt() tests passed!\n")

def test_format_prompt_safe():
    """Test format_prompt_safe function."""
    print("\n" + "="*60)
    print("TEST 2: format_prompt_safe()")
    print("="*60)
    
    # Test 2.1: Valid input
    result = format_prompt_safe("Hello {name}", name="Dave")
    expected = "Hello Dave"
    assert result == expected
    print("Test 2.1 passed: Valid input")
    
    # Test 2.2: Missing variable (should raise ValueError)
    try:
        format_prompt_safe("Hello {name}", age=30)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing required variable" in str(e)
        print("Test 2.2 passed: Missing variable raises error")
    
    # Test 2.3: All variables provided
    result = format_prompt_safe("Hi {x} and {y}", x="A", y="B")
    expected = "Hi A and B"
    assert result == expected
    print("Test 2.3 passed: Multiple variables")
    
    print("All format_prompt_safe() tests passed!\n")


def test_format_prompt_validated():
    """Test format_prompt_validated function."""
    print("\n" + "="*60)
    print("TEST 3: format_prompt_validated()")
    print("="*60)
    
    # Test 3.1: Valid input
    result = format_prompt_validated("Hello {name}", name="Eve")
    expected = "Hello Eve"
    assert result == expected
    print("Test 3.1 passed: Valid input")
    
    # Test 3.2: Missing variable (should raise ValueError)
    try:
        format_prompt_validated("Hello {name} {age}", name="Frank")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing variables" in str(e)
        print("Test 3.2 passed: Missing variable raises error")
    
    # Test 3.3: Extra variable (should print warning)
    print("  Expected warning below:")
    result = format_prompt_validated("Hello {name}", name="Grace", age=40, city="LA")
    expected = "Hello Grace"
    assert result == expected
    print("Test 3.3 passed: Extra variables show warning")
    
    # Test 3.4: Perfect match
    result = format_prompt_validated("Hi {x} and {y}", x="1", y="2")
    expected = "Hi 1 and 2"
    assert result == expected
    print("Test 3.4 passed: Perfect variable match")
    
    print("All format_prompt_validated() tests passed!\n")

def test_prompt_template():
    """Test PromptTemplate class - All methods and edge cases."""
    print("\n" + "="*70)
    print("TEST 1: PromptTemplate Class")
    print("="*70)
    # Test 1.1: Basic initialization and format
    print("\n[Test 1.1] Basic initialzation and format")
    template = PromptTemplate("Hello {name}, welcome to {palce}!")
    result = template.format(name="Kazi", place="Dhaka")
    expected = "Hello Kazi, welcome to Dhaka!"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"Input: name='Alice', place='Paris'")
    print(f"Output: {result}")
    # Test 1.2: Extract variables
    print("\n[Test 1.2] Extract variables")
    vars_list = template.get_variables()
    assert vars_list == ["name", "place"], f"Expected ['name', 'place'], got {vars_list}"
    print(f"Variables extracted: {vars_list}")
    
    # Test 1.3: Missing variable - should raise ValueError
    print("\n[Test 1.3] Missing variable validation")    
    try:
        template.format(name="Kazi")
        assert False, "Should have raised ValueError for missing 'place'"
    except ValueError as e:
        assert "Missing required variables" in str(e)
        assert "place" in str(e)
        print(f"Correctly raised error: {e}")
    
    # Test 1.4: Extra variables - should work fine
    print("\n[Test 1.4] Extra variables (should be ignored)")
    result = template.format(name="Charlie", place="London", age=30, city="NYC")
    expected = "Hello Charlie, welcome to London!"
    assert result == expected
    print(f"Extra variables ignored successfully")

    # Test 1.5: Partial application
    print("\n[Test 1.5] Partial application")
    partial = template.partial(name="Kazi")
    result = partial.format(place="Dhaka")
    expected = "Hello David, welcome to Tokyo!"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"Partial template created with name='David'")
    print(f"Completed with place='Tokyo': {result}")

    # Test 1.6: Partial application with multiple variables
    print("\n[Test 1.6] Multi-variable partial")
    template2 = PromptTemplate("User {user} from {city} likes {hobby}")
    partial2 = template2.partial(user="Eve", city="Berlin")
    remaining = partial2.get_variables()
    assert remaining == ["hobby"], f"Expected ['hobby'], got {remaining}"
    result = partial2.format(hobby="coding")
    assert "Eve" in result and "Berlin" in result and "coding" in result
    print(f"Remaining variables: {remaining}")
    print(f"Final result: {result}")

    # Test 1.7: __repr__ method
    print("\n[Test 1.7] String representation")
    repr_str = repr(template)
    assert "PromptTemplate" in repr_str
    assert "name" in repr_str or "place" in repr_str
    print(f" __repr__: {repr_str}")
    
    # Test 1.8: Empty template (no variables)
    print("\n[Test 1.8] Empty template (no variables)")
    empty = PromptTemplate("This is a static message")
    result = empty.format()
    assert result == "This is a static message"
    assert empty.get_variables() == []
    print(f"Static template works: {result}")
    
    # Test 1.9: Single variable
    print("\n[Test 1.9] Single variable template")
    single = PromptTemplate("Welcome {user}!")
    result = single.format(user="Frank")
    assert result == "Welcome Frank!"
    print(f"Single variable: {result}")
    
    # Test 1.10: Duplicate variables in template
    print("\n[Test 1.10] Duplicate variables")
    dup = PromptTemplate("{name} loves {name}")
    result = dup.format(name="Grace")
    assert result == "Grace loves Grace"
    print(f"Duplicate handling: {result}")
    
    print("\n All PromptTemplate tests passed!\n")

def test_few_shot_prompt():
    """Test FewShotPrompt class - All methods and edge cases."""
    print("\n" + "="*70)
    print("TEST 2: FewShotPrompt Class")
    print("="*70)
    
    # Test 2.1: Basic few-shot prompting
    print("\n[Test 2.1] Basic few-shot with 2 examples")
    examples = [
        {"input": "2 + 2", "output": "4"},
        {"input": "5 * 3", "output": "15"}
    ]
    few_shot = FewShotPrompt("Solve these math problems:", examples)
    result = few_shot.format("10 - 3")
    
    assert "Solve these math problems:" in result
    assert "Input: 2 + 2" in result
    assert "Output: 4" in result
    assert "Input: 5 * 3" in result
    assert "Output: 15" in result
    assert "Input: 10 - 3" in result
    assert result.endswith("Output:")
    print("All examples included in prompt")
    print(f"Prompt preview:\n{result[:150]}...")
    
    # Test 2.2: __len__ method
    print("\n[Test 2.2] Length method")
    length = len(few_shot)
    assert length == 2, f"Expected 2 examples, got {length}"
    print(f"Number of examples: {length}")
    
    # Test 2.3: add_example method
    print("\n[Test 2.3] Add new example")
    few_shot.add_example("8 / 2", "4")
    assert len(few_shot) == 3
    result = few_shot.format("1 + 1")
    assert "8 / 2" in result
    assert "Output: 4" in result
    print("Example added successfully")
    print(f"New length: {len(few_shot)}")
    
    # Test 2.4: Custom input/output keys
    print("\n[Test 2.4] Custom key names")
    custom_examples = [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"}
    ]
    custom_shot = FewShotPrompt(
        "Answer these questions:",
        custom_examples,
        input_key="question",
        output_key="answer"
    )
    result = custom_shot.format("What is DL?")
    assert "question" in result.lower() or "Question" in result
    assert "answer" in result.lower() or "Answer" in result
    assert "Artificial Intelligence" in result
    print("Custom keys work correctly")
    
    # Test 2.5: Custom separator
    print("\n[Test 2.5] Custom separator")
    custom_sep = FewShotPrompt(
        "Examples:",
        [{"input": "a", "output": "A"}],
        separator="\n---\n"
    )
    result = custom_sep.format("b")
    assert "\n---\n" in result
    print("Custom separator applied")
    
    # Test 2.6: Empty examples list
    print("\n[Test 2.6] Empty examples list")
    empty_shot = FewShotPrompt("Task:", [])
    result = empty_shot.format("Do something")
    assert "Task:" in result
    assert "Do something" in result
    assert len(empty_shot) == 0
    print("Empty examples handled correctly")
    
    # Test 2.7: Single example
    print("\n[Test 2.7] Single example")
    single_shot = FewShotPrompt("Example:", [{"input": "x", "output": "y"}])
    result = single_shot.format("z")
    assert len(single_shot) == 1
    assert "Input: x" in result
    print("Single example works")
    
    # Test 2.8: Long instruction
    print("\n[Test 2.8] Long instruction text")
    long_instruction = "This is a very long instruction that explains the task in detail and provides context."
    long_shot = FewShotPrompt(long_instruction, examples)
    result = long_shot.format("test")
    assert long_instruction in result
    print("Long instruction preserved")
    
    print("\n All FewShotPrompt tests passed!\n")


def test_chain_of_thought_prompt():
    """Test ChainOfThoughtPrompt class - All methods and edge cases."""
    print("\n" + "="*70)
    print("TEST 3: ChainOfThoughtPrompt Class")
    print("="*70)
    
    # Test 3.1: With thinking prompt (default)
    print("\n[Test 3.1] Default CoT with thinking prompt")
    cot = ChainOfThoughtPrompt("Solve this problem:")
    result = cot.format("What is 2 + 2?")
    
    assert "Solve this problem:" in result
    assert "Question: What is 2 + 2?" in result
    assert "Let's think step by step:" in result
    print("All components present")
    print(f"Prompt:\n{result}")
    
    # Test 3.2: Without thinking prompt
    print("\n[Test 3.2] CoT without thinking prompt")
    cot_no_thinking = ChainOfThoughtPrompt(
        "Answer this:",
        add_thinking_prompt=False
    )
    result = cot_no_thinking.format("Calculate 5 * 5")
    
    assert "Answer this:" in result
    assert "Question: Calculate 5 * 5" in result
    assert "Let's think step by step" not in result
    print("Thinking prompt not included")
    print(f"Prompt:\n{result}")
    
    # Test 3.3: Long question
    print("\n[Test 3.3] Long question text")
    long_question = "If a train travels at 60 mph for 2 hours, then slows to 40 mph for 1 hour, how far does it travel in total?"
    result = cot.format(long_question)
    assert long_question in result
    assert "Let's think step by step:" in result
    print("Long question preserved")
    
    # Test 3.4: Question with special characters
    print("\n[Test 3.4] Question with special characters")
    special_q = "What is 50% of $100?"
    result = cot.format(special_q)
    assert special_q in result
    print("Special characters handled: $, %")
    
    # Test 3.5: Empty question
    print("\n[Test 3.5] Empty question")
    result = cot.format("")
    assert "Question: " in result
    assert "Let's think step by step:" in result
    print("Empty question handled")
    
    # Test 3.6: Multiline question
    print("\n[Test 3.6] Multiline question")
    multiline_q = "Given:\n- x = 5\n- y = 10\nWhat is x + y?"
    result = cot.format(multiline_q)
    assert multiline_q in result
    print("Multiline question preserved")
    
    # Test 3.7: Numeric question
    print("\n[Test 3.7] Pure numeric question")
    result = cot.format("123 + 456")
    assert "123 + 456" in result
    print("Numeric question works")
    
    print("\n All ChainOfThoughtPrompt tests passed!\n")


def test_conversation_template():
    """Test ConversationTemplate class - All methods and edge cases."""
    print("\n" + "="*70)
    print("TEST 4: ConversationTemplate Class")
    print("="*70)
    
    # Test 4.1: Basic conversation with system prompt
    print("\n[Test 4.1] Basic conversation (simple format)")
    conv = ConversationTemplate(system_prompt="You are a helpful assistant.")
    conv.add_user_message("Hello!")
    conv.add_assistant_message("Hi there! How can I help you?")
    conv.add_user_message("What is AI?")
    
    result = conv.format(format_type="simple")
    assert "System: You are a helpful assistant." in result
    assert "User: Hello!" in result
    assert "Assistant: Hi there! How can I help you?" in result
    assert "User: What is AI?" in result
    print("Simple format works")
    print(f"Conversation:\n{result}")
    
    # Test 4.2: Chat format
    print("\n[Test 4.2] Chat API format")
    chat_result = conv.format(format_type="chat")
    
    assert isinstance(chat_result, list)
    assert len(chat_result) == 4  # 1 system + 3 messages
    assert chat_result[0]["role"] == "system"
    assert chat_result[0]["content"] == "You are a helpful assistant."
    assert chat_result[1]["role"] == "user"
    assert chat_result[1]["content"] == "Hello!"
    assert chat_result[2]["role"] == "assistant"
    print("Chat format returns list of dicts")
    print(f"Number of messages: {len(chat_result)}")
    
    # Test 4.3: __len__ method
    print("\n[Test 4.3] Length method")
    length = len(conv)
    assert length == 3, f"Expected 3 messages, got {length}"
    print(f"Message count (excluding system): {length}")
    
    # Test 4.4: add_message method
    print("\n[Test 4.4] Add custom role message")
    conv.add_message("system", "Additional context")
    assert len(conv) == 4
    print("Custom message added")
    
    # Test 4.5: clear method
    print("\n[Test 4.5] Clear messages")
    conv.clear()
    assert len(conv) == 0
    result = conv.format(format_type="simple")
    assert "System: You are a helpful assistant." in result
    assert "User:" not in result
    print("Messages cleared, system prompt retained")
    
    # Test 4.6: No system prompt
    print("\n[Test 4.6] Conversation without system prompt")
    conv_no_sys = ConversationTemplate()
    conv_no_sys.add_user_message("Hi")
    conv_no_sys.add_assistant_message("Hello")
    
    result = conv_no_sys.format(format_type="simple")
    assert "System:" not in result
    assert "User: Hi" in result
    print("No system prompt works")
    
    # Test 4.7: Invalid format type
    print("\n[Test 4.7] Invalid format type")
    try:
        conv_no_sys.format(format_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown format type" in str(e)
        print(f"Error raised correctly: {e}")
    
    # Test 4.8: Empty conversation
    print("\n[Test 4.8] Empty conversation")
    empty_conv = ConversationTemplate()
    assert len(empty_conv) == 0
    result = empty_conv.format(format_type="simple")
    assert result == ""
    print("Empty conversation handled")
    
    # Test 4.9: Long messages
    print("\n[Test 4.9] Long message content")
    long_conv = ConversationTemplate()
    long_msg = "This is a very long message " * 20
    long_conv.add_user_message(long_msg)
    result = long_conv.format(format_type="simple")
    assert long_msg in result
    print("Long messages preserved")
    
    # Test 4.10: Multiple consecutive messages from same role
    print("\n[Test 4.10] Multiple messages from same role")
    multi_conv = ConversationTemplate()
    multi_conv.add_user_message("First question")
    multi_conv.add_user_message("Second question")
    multi_conv.add_user_message("Third question")
    assert len(multi_conv) == 3
    result = multi_conv.format(format_type="simple")
    assert "First question" in result
    assert "Third question" in result
    print("Multiple same-role messages work")
    
    # Test 4.11: Special characters in messages
    print("\n[Test 4.11] Special characters in messages")
    special_conv = ConversationTemplate()
    special_conv.add_user_message("What's 50% of $100?")
    special_conv.add_assistant_message("It's $50!")
    result = special_conv.format(format_type="simple")
    assert "$" in result
    assert "%" in result
    print("Special characters preserved")
    
    print("\n All ConversationTemplate tests passed!\n")


if __name__ == "__main__":
    test_format_prompt()
    test_format_prompt_safe()
    test_format_prompt_validated()
    test_prompt_template()
    test_few_shot_prompt()
    test_chain_of_thought_prompt()
    test_conversation_template()
