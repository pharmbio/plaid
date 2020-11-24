# multiplate-designer

**PLAID** (Plates created using Artificial Intelligence Design) is a flexible
constraint-programming model representing the Plate Layout Design
problem. **PLAID** was developed with the goal of helping
researchers plan well-designed experiments by creating a robust
microplate layout and thus reducing the rate of (partial) microplate
rejection.<sup>[1](#RefModRef2020)</sup>

**PLAID**  is easy and straightforward to use. The current
model guarantees the following constraints:

* The outermost rows and columns can be left empty in order to reduce
errors due to the edge effect.
* For each compound or combination thereof:
  - all concentration levels of a given replica appear on the same
  plate.
  - each concentration level appears on a different row and column.
* If possible, the replicated compounds and combinations appear on a
different plate.
* For each type of control and concentration, the difference in number
between plates is at most 1.
* Controls of the same kind are separated by at least 1 well in any
direction.
* If empty wells are allowed (other than those used to reduce the edge
effect), they are placed as close to the border as possible.

Using **PLAID** does not require any programming knowledge.
Users just need to write down the necessary information such as number
of compounds, combinations, controls, etc in a simple text file and
click run!

The output is a list in .csv format containing plate ID, well, content
(compound, combination or control), concentration, and latex name,
meant to be used as input for automatic tools. (TODO: visualize the
layout using latex)

We believe **PLAID** is the first attempt to use constraint
programming to design microplate layouts. Due to the use of MiniZinc,<sup>[2](#RefMiniZinc)</sup>
a high level constraint modelling language, **PLAID** is
highly customizable.



## Table of Contents
* [Installation and Usage](#installation)
  - [Using the MiniZinc IDE](#minizinc-ide)
  - [Using the command line](#command-line)
* [References](#references)


<a name="installation"></a>
## Installation and Usage

* Download and install [MiniZinc](https://www.minizinc.org/).
* Clone this repo OR download plate-design.mzn and empty-file.dzn.
* Fill in the specific details of your experiment in a
empty-file.dzn. Alternatively, you can download and modify any of the example .dzn
files. (TODO: add an example input file in JSON)


<a name="minizinc-ide"></a>
### Using the MiniZinc IDE
* Open both plate-design.mzn and your .dzn file (if you have one) using the MiniZinc IDE
* In the dropdown menu called "Solver configuration", located in the
top middle area of the MiniZinc IDE, select
Gecode<sup>[3](#RefGecode)</sup> as solver (Do not select Gecode Gist,
which is used for debugging).
* Click the "Run" button, located to the
left of "Solver configuration".
* In the popup window, you can either select a data file (you can only
see those that are open in the IDE) or enter all parameters by hand.
Note that you need to scroll down inside the popup to be able to type
in all the values.
* Optional: you can change the random seed to obtain a different layout for the
same input data under "MiniZinc/Solver Configuration/Show
configuration editor...". 


<a name="command-line"></a>
### Using the command line
TODO: add more info here

$ minizinc --solver Gecode plate-design.mzn pl-example01.dzn

<a name="references"></a>
## References

<a name="RefModRef2020">1</a>: M. A. Francisco Rodríguez, and O. Spjuth. *A Constraint
Programming Approach to Microplate Layout Design* In: J. Espasa and N.
Dang (editors), Proceedings of ModRef 2020, the 19th International
Workshop on Constraint Modelling and Reformulation, held at CP 2020,
September 2020.
[[PDF](https://modref.github.io/papers/ModRef2020_A%20Constraint%20Programming%20Approach%20to%20Microplate%20Layout%20Design.pdf)]
[[Slides](https://modref.github.io/slides/ModRef2020_Slides_A%20Constraint%20Programming%20Approach%20to%20Microplate%20Layout%20Design.pdf)]
[[Video](https://www.youtube.com/watch?v=naddH2TQIjE&ab_channel=CP2020)]


<a name="RefMiniZinc">2</a>: Nethercote, N., Stuckey, P.J., Becket, R., Brand, S., Duck, G.J.,
Tack, G.: MiniZinc:Towards a Standard CP Modelling Language. In:
Bessière, C. (ed.) Principles andPractice of Constraint Programming –
CP 2007. pp. 529–543. Lecture Notes inComputer Science, Springer,
Berlin, Heidelberg (2007)


<a name="RefGecode">3</a>: Gecode Team: Gecode: Generic constraint development environment
(2019), available from http://www.gecode.org
