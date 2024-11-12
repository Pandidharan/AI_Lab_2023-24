# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE:     14/10/24                                                                       
### REGISTER NUMBER : 212222040111
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
likes(john,X):-
 food(X).
eats(bill,X):-
 eats(sue,X).
eats(Y,X):-
 food(X).
eats(bill,peanuts).
food(apple).
food(chicken).
food(peanuts).
```

### Output:
![image](https://github.com/user-attachments/assets/edc254bf-61d4-4007-b89d-5c6e3d6229de)

### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:
```
likes(steve,X):-
 easycourse(X).
hard(sciencecourse).
easycourse(X):-
 course(X,dept(havefun)).
course(bk301,dept(havefun)).
```

### Output:
![image](https://github.com/user-attachments/assets/d754e5d3-d040-4d09-bff8-55f4019889e4)
### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 

### Program:
```
 american(X),
 weapon(Y),
 hostile(Z),
 sells(X,Y,Z).
weapon(Y):-
 missile(Y).
hostile(Z):-
 enemy(Z,X).
sells(west,Y,nano):-
 missile(Y),
 owns(nano,Y).
missile(m).
owns(nano,m).
enemy(nano,america).
american(west). 
```

### Output:
![image](https://github.com/user-attachments/assets/1fbdfa64-c390-4a6f-9a89-f0d88da00d6a)


### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
