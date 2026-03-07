import re 
from typing import Dict, List, Optional, Any 

def format_prompt(template: str, **kwargs) -> str:
    """
    Fill in a prompt template with variables.

    Args:
        template: String with placeholders like {variable name}
        **kwargs: Variable values to substitute
        
    Returns:
        Formatted string with placeholders replaced
    
    Example:
        >>> template = "You are a {role}. Answer: {question}"
        >>> format_prompt(template, role="assistant", question="What is AI?")
        'You are a assistant. Answer: What is AI?'
    """
    return template.format(**kwargs)

def format_prompt_safe(template: str, **kwargs) -> str:
    """
    Format with error handling for missing variables.
    
    Args:
        template: String with placeholders
        **kwargs: Variable values
        
    Returns:
        Formatted string
        
    Raises:
        ValueError: If required variables are missing
        
    Example:
        >>> format_prompt_safe("Hello {name}", age=25)
        ValueError: Missing required variable: 'name'
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable: {e}")
    
def format_prompt_validated(template: str, **kwargs) -> str:
    """
    Format with full validation - checks for missing and extra variables.
    
    Args:
        template: String with placeholders
        **kwargs: Variable values
        
    Returns:
        Formatted string
        
    Raises:
        ValueError: If required variables are missing
        
    Warnings:
        Prints warning if extra unused variables are provided
        
    Example:
        >>> template = "Hello {name}, you are {age} years old"
        >>> format_prompt_validated(template, name="Alice", age=25, city="NYC")
        Warning: Unused variables: {'city'}
        'Hello Alice, you are 25 years old'
    """
    # Find all placeholders in template
    placeholders = set(re.findall(r'\{(\w+)\}', template))
    provided = set(kwargs.keys())
    
    # Check for missing variables
    missing = placeholders - provided
    if missing:
        raise ValueError(f"Missing variables: {missing}")
    
    # Check for extra variables
    extra = provided - placeholders
    if extra:
        print(f"Warning: Unused variables: {extra}")
    
    return template.format(**kwargs)

class PromptTemplate:
    """
    Reusable prompt template with validation and partial application.
    
    Example:
        >>> template = PromptTemplate("You are a {role}. Answer: {question}")
        >>> prompt = template.format(role="assistant", question="What is AI?")
        >>> 
        >>> # Partial application
        >>> assistant_template = template.partial(role="helpful assistant")
        >>> prompt = assistant_template.format(question="What is ML?")
    """
    def __init__(self, template: str, required_vars: Optional[List[str]] = None):
        """
        Initialize prompt template.
        
        Args:
            template: String with placeholders
            required_vars: List of required variable names (auto-detected if None)
        """
        self.template = template
        self.required_vars = required_vars or self._extract_vars()

    def _extract_vars(self) -> List[str]:
        """Extract variable names from template using regex."""
        return re.findall(r'\{(\w+)\}', self.template)
    

    def format(self, **kwargs) ->str:
         """
        Format the template with validation.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted string
            
        Raises:
            ValueError: If required variables are missing
        """ 
         missing = set(self.required_vars) - set(kwargs.keys())
         if missing:
            raise ValueError(f"Missing required variables: {missing}")
         return self.template.format(**kwargs)
    
    def partial(self, **kwargs) -> 'PromptTemplate':
        """
        Create a new template with some variables pre-filled.
        
        Args:
            **kwargs: Variables to pre-fill
            
        Returns:
            New PromptTemplate with some variables filled in
            
        Example:
            >>> base = PromptTemplate("Hello {name}, you are {age}")
            >>> greeting = base.partial(name="Alice")
            >>> greeting.format(age=25)
            'Hello Alice, you are 25'
        """
        # Fill in the provided variables
        partial_template = self.template
        for key, value in kwargs.items():
            partial_template = partial_template.replace(f"{{{key}}}", str(value))
        
        # Determine remaining variables
        remaining_vars = [v for v in self.required_vars if v not in kwargs]
        return PromptTemplate(partial_template, remaining_vars)
    
    def get_variables(self) -> List[str]:
        """Get list of all variables in the template."""
        return self.required_vars.copy()
    
    def __repr__(self) -> str:
        """String representation of the template."""
        return f"PromptTemplate(vars={self.required_vars})"
class FewShotPrompt:
    """
    Template for few-shot learning with examples.
    
    Example:
        >>> examples = [
        ...     {"input": "2 + 2", "output": "4"},
        ...     {"input": "5 * 3", "output": "15"}
        ... ]
        >>> prompt = FewShotPrompt("Solve these math problems:", examples)
        >>> result = prompt.format("10 - 3")
        >>> print(result)
        Solve these math problems:
        
        Input: 2 + 2
        Output: 4
        
        Input: 5 * 3
        Output: 15
        
        Input: 10 - 3
        Output:
    """

    def __init__(
            self, 
            instruction:str, 
            examples:List[Dict[str, str]], 
            input_key:str="input", 
            output_key="output", 
            separator:str="\n\n"
    ):
        """
        Initialize few-shot prompt template.
        
        Args:
            instruction: System instruction or task description
            examples: List of example dictionaries with input/output pairs
            input_key: Key name for input in examples (default: "input")
            output_key: Key name for output in examples (default: "output")
            separator: String to separate examples (default: "\n\n")
        """
        self.instruction = instruction
        self.examples = examples
        self.input_key = input_key
        self.output_key = output_key
        self.separator = separator

    def format(self, query: str) -> str:
        """
        Build few-shot prompt with examples and new query.
        
        Args:
            query: New input to process
            
        Returns:
            Complete prompt with instruction, examples, and query
        """
        prompt = self.instruction + self.separator
        
        # Add examples
        for ex in self.examples:
            prompt += f"Input: {ex[self.input_key]}\n"
            prompt += f"Output: {ex[self.output_key]}{self.separator}"
        
        # Add actual query
        prompt += f"Input: {query}\nOutput:"
        return prompt
    def add_example(self, input_text: str, output_text: str) -> None:
        """Add a new example to the template."""    
        self.examples.append({
            self.input_key: input_text,
            self.output_key: output_text
        })
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

class ChainOfThoughtPrompt:
    """
    Template for chain-of-thought prompting.
    
    Example:
        >>> cot = ChainOfThoughtPrompt("Solve this math problem step by step:")
        >>> prompt = cot.format("If John has 5 apples and gives 2 to Mary, how many does he have?")
    """
    def __init__(self, instruction: str, add_thinking_prompt: bool = True):
        """
        Initialize CoT prompt.
        
        Args:
            instruction: Task instruction
            add_thinking_prompt: Whether to add "Let's think step by step"
        """
        self.instruction = instruction
        self.add_thinking_prompt = add_thinking_prompt
    
    def format(self, query: str) -> str:
        """Format query with CoT prompting."""
        prompt = f"{self.instruction}\n\nQuestion: {query}\n"
        
        if self.add_thinking_prompt:
            prompt += "\nLet's think step by step:\n"
        return prompt
class ConversationTemplate:
    """
    Template for multi-turn conversations.
    
    Example:
        >>> conv = ConversationTemplate(system_prompt="You are a helpful assistant.")
        >>> conv.add_user_message("Hello!")
        >>> conv.add_assistant_message("Hi! How can I help?")
        >>> conv.add_user_message("What is AI?")
        >>> prompt = conv.format()
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize conversation template.
        
        Args:
            system_prompt: Optional system message to prepend
        """
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        self.add_message("assistant", content)
    
    def format(self, format_type: str = "simple") -> str:
        """
        Format the conversation.
        
        Args:
            format_type: "simple" or "chat" format
            
        Returns:
            Formatted conversation string
        """
        if format_type == "simple":
            return self._format_simple()
        elif format_type == "chat":
            return self._format_chat()
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
    def _format_simple(self) -> str:
        """Simple text format."""
        lines = []
        if self.system_prompt:
            lines.append(f"System: {self.system_prompt}\n")

        for msg in self.messages:
            lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(lines)
    
    def _format_chat(self) -> List[Dict[str, str]]:
        """Chat API format (returns list of message dicts)."""
        formatted = []
        
        if self.system_prompt:
            formatted.append({"role": "system", "content": self.system_prompt})
        
        formatted.extend(self.messages)
        return formatted
    
    def clear(self) -> None:
        """Clear all messages (keeps system prompt)."""
        self.messages = []
    
    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)        
