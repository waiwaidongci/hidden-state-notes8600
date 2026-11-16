import math
import numpy as np
from scipy.special import logsumexp
from .hmm import ImpossibleSequence,log_probabilities

def _check_observations(obs,B):
 arr=np.asarray(obs)
 if arr.ndim!=1 or not 1<=len(arr)<=200 or not np.issubdtype(arr.dtype,np.integer) or np.issubdtype(arr.dtype,np.bool_) or (arr<0).any() or (arr>=B.shape[1]).any():
  raise ValueError('观测须1..200个合法整数码索引')
 return arr.astype(int)

def e_step(observations,pi,transition,emission):
 p=np.asarray(pi,dtype=float);A=np.asarray(transition,dtype=float);B=np.asarray(emission,dtype=float);obs=_check_observations(observations,B)
 T=len(obs);K=len(p)
 lp=log_probabilities(p);lA=log_probabilities(A);lB=log_probabilities(B)
 alpha=np.full((T,K),-np.inf);alpha[0]=lp+lB[:,obs[0]]
 if not np.isfinite(alpha[0]).any():raise ImpossibleSequence('不可能序列：第1码在当前模型下概率为0')
 for t in range(1,T):
  alpha[t]=logsumexp(alpha[t-1,:,None]+lA,axis=0)+lB[:,obs[t]]
  if not np.isfinite(alpha[t]).any():raise ImpossibleSequence(f'不可能序列：到第{t+1}码所有路径概率为0')
 log_likelihood=float(logsumexp(alpha[-1]))
 beta=np.zeros((T,K))
 for t in range(T-2,-1,-1):
  beta[t]=logsumexp(lA+lB[:,obs[t+1]][None,:]+beta[t+1][None,:],axis=1)
 with np.errstate(under='ignore'):
  gamma=np.exp(alpha+beta-log_likelihood)
 gamma/=gamma.sum(axis=1,keepdims=True)
 xi_sum=np.zeros((K,K))
 for t in range(T-1):
  with np.errstate(under='ignore'):
   xi_sum+=np.exp(alpha[t,:,None]+lA+lB[:,obs[t+1]][None,:]+beta[t+1][None,:]-log_likelihood)
 emission_counts=np.zeros((K,B.shape[1]))
 for t,symbol in enumerate(obs):emission_counts[:,symbol]+=gamma[t]
 return {'log_likelihood':log_likelihood,'gamma':gamma,'xi_sum':xi_sum,'pi_counts':gamma[0].copy(),'emission_counts':emission_counts}

def sequence_log_likelihood(observations,pi,transition,emission):
 p=np.asarray(pi,dtype=float);A=np.asarray(transition,dtype=float);B=np.asarray(emission,dtype=float);obs=_check_observations(observations,B)
 lp=log_probabilities(p);lA=log_probabilities(A);lB=log_probabilities(B)
 alpha=lp+lB[:,obs[0]]
 if not np.isfinite(alpha).any():raise ImpossibleSequence('不可能序列：第1码在当前模型下概率为0')
 for t in range(1,len(obs)):
  alpha=logsumexp(alpha[:,None]+lA,axis=0)+lB[:,obs[t]]
  if not np.isfinite(alpha).any():raise ImpossibleSequence(f'不可能序列：到第{t+1}码所有路径概率为0')
 return float(logsumexp(alpha))

def _zero_preserving_row(counts,previous_row):
 total=float(np.asarray(counts,dtype=float).sum())
 if total<=0.0:return [float(v) for v in previous_row],False
 updated=np.asarray(counts,dtype=float)/total
 return [float(v) for v in updated],True

def m_step(pi_counts,transition_counts,emission_counts,previous_parameters):
 pi_c=np.asarray(pi_counts,dtype=float);A_c=np.asarray(transition_counts,dtype=float);B_c=np.asarray(emission_counts,dtype=float)
 prev_pi=previous_parameters['pi'];prev_A=previous_parameters['transition'];prev_B=previous_parameters['emission']
 pi,_=_zero_preserving_row(pi_c,prev_pi)
 transition=[]
 for i,row in enumerate(A_c):
  updated,used=_zero_preserving_row(row,prev_A[i])
  transition.append(updated)
 emission=[]
 for i,row in enumerate(B_c):
  updated,_=_zero_preserving_row(row,prev_B[i])
  emission.append(updated)
 return {'pi':pi,'transition':transition,'emission':emission,'zero_policy':'Structural zeros stay zero; rows without expected counts keep the previous row; no epsilon floors.'}

def batch_expected_counts(traces_integer,parameters):
 pi=parameters['pi'];A=parameters['transition'];B=parameters['emission']
 K=len(pi);M=np.asarray(B).shape[1]
 pi_sum=np.zeros(K);A_sum=np.zeros((K,K));B_sum=np.zeros((K,M));log_likelihoods=[];token_count=0
 for obs in traces_integer:
  stat=e_step(obs,pi,A,B)
  pi_sum+=stat['pi_counts'];A_sum+=stat['xi_sum'];B_sum+=stat['emission_counts']
  log_likelihoods.append(stat['log_likelihood']);token_count+=len(obs)
 return {'pi_counts':pi_sum.tolist(),'transition_counts':A_sum.tolist(),'emission_counts':B_sum.tolist(),'log_likelihood':float(math.fsum(log_likelihoods)),'token_count':token_count}

def baum_welch_round(traces_integer,parameters):
 counts=batch_expected_counts(traces_integer,parameters)
 updated=m_step(counts['pi_counts'],counts['transition_counts'],counts['emission_counts'],parameters)
 new_log_likelihood=float(math.fsum(sequence_log_likelihood(obs,updated['pi'],updated['transition'],updated['emission']) for obs in traces_integer))
 return {'parameters':updated,'previous_log_likelihood':counts['log_likelihood'],'new_log_likelihood':new_log_likelihood,'counts':counts}
