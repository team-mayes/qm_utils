%chk=oxane.chk 
# opt(modredundant) m062x/6-31G(d) scf=xqc nosymm
example holding ring constrained

1  3
 C          1.32000        -0.00000        -0.23300
 C          0.66000         1.14300         0.23300
 C         -0.66000         1.14300        -0.23300
 C         -1.32000        -0.00000         0.23300
 C         -0.66000        -1.14300        -0.23300
 O          0.66000        -1.14300         0.23300

D   1    2    3    4 F 
D   2    3    4    5 F 
D   3    4    5    6 F 
D   4    5    6    1 F
