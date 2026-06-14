**Evaluating computations**

I'm considering whether I can compute chi_iota for round 0 using only the outputs needed for round 1 theta. However, to get a complete round 1, I actually need all 25 outputs. It's interesting to think if I might be able to skip zeroing the state; I'll have to think that through. It seems like there are a lot of interconnected elements here, so I should probably keep everything in mind to make sure I don't miss anything.
**Exploring implementation options**

I’m considering implementing a combined version that handles round22 and the last one, storing only the c/diag information. There’s definitely potential for improvement here! Let’s explore this idea further and possibly see how we can make it work effectively. I think this could make a real difference in the process, and I'm excited to dive into the details and see how best to implement it.