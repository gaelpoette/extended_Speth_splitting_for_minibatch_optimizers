# extended_Speth_splitting_for_minibatch_optimizers
Code for the numerical experiments of paper entitled 
"NUMERICAL SPLITTING SCHEMES AS THE CORNERSTONE FOR MINI-BATCH OPTIMIZATION: ON THE IMPORTANCE OF WELL-BALANCED METHODS" 
by Bilel Bensaid, Gaël Poëtte & Rodolphe Turpault

# The 1D test-cases on which the difficulties relative to being interpolating or not are all in 
1D_non_interpolating_benchmarks

#The GD (respectively Momentum and Adam) based mini-batch optimizers are in GD_based (Momentum_based and Adam_based)

# To run RR-GD on the 8 benchmarks of the paper:
python rr_gd.py
# To run eS-GD:
python eS_GD.py
# To run the low memory version of eS-GD:
python lm_eS_GD.py
# To run eS-GD with random reshuffle mini-batching (instead of deterministic)
python rr_eS_GD.py

# To run S-Momentum
python s_mom.py 
# To run eS-Momentum with the splitting on the full flow F
python eS_mom_F.py 
# To run eS-Momentum with only the aggregator on the gradient
python eS_mom.py 

 # To run RR-Adam
python Adam.py 
# To run full batch Adam
python Adam_fb.py 
# To run eS-Adam with only the aggregator on the gradient BUT with the need to clip the squared component of the gradient
python eS_Adam.py 


