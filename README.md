# Reinfier & Reintrainer
Reinfier is a general verifier and interpreter for deep reinforcement learning.  
The verifer part combines the formal verification of deep neural network with bounded model checking algorithm and k-induction algorithm to verify the properties of deep reinforcement learning or give counterexamples. 


The interpreter part

Source code is available at [Reinfier](https://github.com/Kurayuri/Reinfier).
## Installation

### Reinfier
Required Python>=3.10.

Reinfier are recommended to install with conda environment.

```shell
# Create a conda environment named reinfier
conda create -n reinfier python

# Activate the conda environment:
conda activate reinfier
```

Install from PyPI:  
```shell
pip install reinfier
```
Or install from source:  
```shell
# Clone the repo
git clone https://github.com/Kurayuri/Reinfier.git
cd Reinfier

# Install in editable mode
pip install -e .
```

### Verification Backend
#### DNNV
Reinfier takes [DNNV](https://github.com/dlshriver/dnnv) as the external DNN verification framework now, which requrires verifiers of DNN ([Reluplex](https://github.com/guykatzz/ReluplexCav2017), [planet](https://github.com/progirep/planet), [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl), [Neurify](https://github.com/tcwangshiqi-columbia/Neurify), [ERAN](https://github.com/eth-sri/eran), [BaB](https://github.com/oval-group/PLNN-verification), [marabou](https://github.com/NeuralNetworkVerification/Marabou), [nnenum](https://github.com/stanleybak/nnenum), [verinet](https://vas.doc.ic.ac.uk/software/neural/)).  

For DRL verification, Reinfier now supports Marabou, Neurify, nnenum and Planet well. For DNN verifcation, Reinfier supports ones as same as DNNV.

Instal DNNV from source:  
```shell
pip install git+https://github.com/dlshriver/DNNV.git@develop
```
Or from docker:  
```shell
# Pull DNNV image
docker pull dlshriver/dnnv:latest

# Create and start a conatiner named dnnv, and keep it running in background
docker run --name dnnv dlshriver/dnnv:latest tail -f /dev/null

# Start bash in the container
docker exec -it dnnv bash
```

To install any of the supported verifiers, run:
```shell
dnnv_manage install reluplex planet mipverify neurify eran bab marabou nnenum verinet
```

#### Marabou
Reinfier can also take [Marabou](https://github.com/NeuralNetworkVerification/Marabou) directly.

```shell
pip install maraboupy
```


### Test Installation
Usage sample files to test:  
```python
import reinfier as rf

network, property = rf.res.get_example()

print(rf.verify(network, property))
```
The result should be:
```python
(False, 2, an instance of <numpy.ndarray>)
```
which means the property is False (SAT, Invalid) with verification depth is 2, and a violation (counterexample) is given.

## Usage
### Verification
A **DRLP** object storing a property in DRLP format and an **NN** object storing an ONNX DNN are required for a basic DRL verification query in Reinfier.

```python
import reinfier as rf

network = rf.NN("/path/to/ONNX/file")
# or
network = rf.NN(ONNX_object)

property = rf.DRLP("/path/to/DRLP/file")
# or
property = rf.DRLP(DRLP_str)

# Verify API (default k-induction algorithm, Recommended)
result = rf.verify(network, property)
# or
# k-induction algorithm 
result = rf.k_induction(network, property) 
# or
# Bounded model checking algorithm
result = rf.bmc(network, property) 
# or
# Reachability analysis algorithm
result = rf.reachability(network, property)
```

### Interpretation
```python
# Set search keyword arguments
kwargs = {
    "a": {"lower_bound": -0.7,
          "upper_bound": -0.3,
          "precise": 0.02,
          "method": "linear", },
    "b": {"lower_bound": 1,
          "upper_bound": 20,
          "precise": 0.1,
          "method": "binary", },
}
# Search breakpoints
breakpoints = rf.search_breakpoints(network, property, kwargs)

# Analyze breakpoints
inline_breakpoints, inline_breaklines = rf.analyze_breakpoints(breakpoints)

# Answer interpretability problems
# Importance Analysis
result = rf.interpreter.answer_importance_analysis(inline_breakpoints)
# Sensitivity Analysis
result = rf.interpreter.answer_sensitivity_analysis(inline_breakpoints)
# Intuitiveness Examination
result = rf.interpreter.answer_intuitiveness_examination(inline_breakpoints)
# Counterfactual Explanation
result = rf.interpreter.answer_counterfactual_explanation(inline_breakpoints)
# Decision Boundary
rf.interpreter.draw_decision_boundary(inline_breakpoints)
```

### Training
```python
reintrainer = rf.Reintrainer(
    [list_of_desired_DRLP_properties],
    train_api)

reintrainer.train(max_iterations)

```

## DRLP
DRLP, i.e. Deep Reinforcement Learning Property, is a Pyhton-embedded DSL to describe property of DRL.
### Reserved Keywords
| Parameter                | Variable Keyword |       Type      |
|--------------------------|:----------------:|:---------------:|
| Input of NN $x$          |        $x$       | $numpy.ndarray$ |
| Output of NN $y$         |        $y$       | $numpy.ndarray$ |
| Input size of NN         |     $x\_size$    |      $int$      |
| Output size of NN        |     $y\_size$    |      $int$      |
| Verification depths $k$  |        $k$       |      $int$      |
### Example
```python
_a = [0,1]

@Pre
[[-1]*2]*k <= x <= [[1]*2]*k

[a]*2 == x[0]

for i in range(0,k-1):
    Implies(y[i] > [0], x[i]+0.5 >= x[i+1] >= x[i])
    Implies(y[i] <= [0], x[i]-0.5 <= x[i+1] <= x[i])

@Exp
y >= [[-2]]*k
```
Such DRLP text describe an Environment and an Agent:  
Becouse of Initial state 𝐼 consists of two situtions in fact, such DRLP describes two concrete properties.
1. State boundary S: Each input value is within $[−1,1]$  
2. Initial state 𝐼: Each input value is $0$ or Each input value is $1$
3. State transition 𝑇: Each input value of the next state increases by at most $0.5$ when output is greater than $0$, and each input value of the next state decreases by at most $0.5$ when output is not greater than $0$
4. Other constraints 𝐶: None
5. Post-condition 𝑄: Output is always not less than $-2$ 

### Defination

The dinfination of DRLP:  
```BNF
<drlp> ::= (<statements> NEWLINE '@Pre' NEWLINE)
            <io_size_assign> NEWLINE <statements> NEWLINE 
            '@Exp' NEWLINE <statements>

<io_size_assign> ::= ''
   |  <io_size_assign> NEWLINE <io_size_id> '=' <int>
   
<io_size_id> = 'x_size' | 'y_size'

<statements> ::= ''
    | <statements> NEWLINE <statement>

<statement> ::= <compound_stmt> | <simple_stmts>

<compound_stmt> ::= <for_stmt> | <with_stmt>

<for_stmt> :: = 'for' <id> 'in' <range_type> <for_range> ':' <block>

<with_stmt> :: = 'with'  <range_type> ':' <block>

<block> ::= NEWLINE INDENT <statements> DEDENT
    | <simple_stmts>

<range_type> ::= 'range' | 'orange'

<for_range> ::= '('<int>')'
    | '('<int> ',' <int> ')'
    | '('<int> ',' <int> ',' <int>')'

<simple_stmts> ::= ""
    | <simple_stmts> NEWLINE <simple_stmt>

<simple_stmt> ::= <call_stmt> | <expr>

<call_stmt> ::= 'Impiles' '(' <expr> ',' <expr> ')'
    | 'And' '(' <exprs> ')'
    | 'Or' '(' <exprs> ')'

<exprs> ::= <expr> 
    | <exprs> ',' <expr>

<expr> ::= <obj> <comparation>

<comparation> ::= '' 
    | <comparator> <obj> <comparation>

<obj> ::= <constraint> | <io_obj>

<io_obj> ::= <io_id> 
    | <io_id> <subscript>
    | <io_id> <subscript> <subscript>
    
<io_id> ::= 'x' | 'y'

<subscript> ::= '[' <int> ']'
     | '[' <int>':'<int> ']'
     | '[' <int>':'<int> ':'<int>']'

<int> ::= <int_number> 
    | <id> 
    | <int> <operator> <int>

<constraint> :: = <int> 
    | <list>
    | <constraint> <operator> <constraint>

```