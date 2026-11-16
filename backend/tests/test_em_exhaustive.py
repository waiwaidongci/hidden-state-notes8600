"""短序列穷举核对 Baum-Welch 期望计数：对全部状态路径做枚举加权，
与 log 域前后向算出的 gamma / xi / 发射期望计数逐项比较。"""
import itertools,math
import numpy as np
from src.services.training_em import e_step,m_step,sequence_log_likelihood,baum_welch_round,batch_expected_counts
from src.services.hmm import ImpossibleSequence

def brute_force(obs,pi,A,B):
 K=len(pi);T=len(obs);paths=list(itertools.product(range(K),repeat=T))
 weighted=[]
 for path in paths:
  weight=pi[path[0]]*B[path[0],obs[0]]
  for t in range(1,T):weight*=A[path[t-1],path[t]]*B[path[t],obs[t]]
  weighted.append((path,weight))
 log_likelihood=math.log(sum(w for _,w in weighted))
 gamma=np.zeros((T,K));xi=np.zeros((K,K));emission=np.zeros((K,B.shape[1]))
 for path,weight in weighted:
  for t,s in enumerate(path):gamma[t,s]+=weight
  for t in range(T-1):xi[path[t],path[t+1]]+=weight
  for t,symbol in enumerate(obs):emission[path[t],symbol]+=weight
 gamma/=np.exp(log_likelihood);xi/=np.exp(log_likelihood);emission/=np.exp(log_likelihood)
 pi_counts=gamma[0].copy()
 return log_likelihood,gamma,xi,emission,pi_counts

CASES=[
 ([.6,.4],[[.7,.3],[.4,.6]],[[.5,.5],[.3,.7]],[0,1,0]),
 ([.6,.4],[[.7,.3],[.4,.6]],[[.5,.5],[.3,.7]],[1,1,0]),
 ([1.,0.],[[.8,.2],[.4,.6]],[[.6,.4],[.2,.8]],[0,0,1]),
 ([.7,.3],[[1.,0.],[0.,1.]],[[.5,.5],[.5,.5]],[1,0,1]),
 ([.5,.5],[[.9,.1],[.2,.8]],[[.7,.3],[.4,.6]],[0,0,0]),
]

def test_expected_counts_match_exhaustive_paths():
 for pi,A,B,obs in CASES:
  A=np.array(A);B=np.array(B)
  stat=e_step(obs,pi,A,B)
  log_likelihood,gamma,xi,emission,pi_counts=brute_force(np.array(obs),np.array(pi),A,B)
  assert math.isclose(stat['log_likelihood'],log_likelihood,rel_tol=1e-12,abs_tol=1e-12)
  assert np.allclose(stat['gamma'],gamma,atol=1e-12),obs
  assert np.allclose(stat['xi_sum'],xi,atol=1e-12),obs
  assert np.allclose(stat['emission_counts'],emission,atol=1e-12),obs
  assert np.allclose(stat['pi_counts'],pi_counts,atol=1e-12)
  assert np.allclose(stat['gamma'].sum(axis=1),1.0,atol=1e-12)
  assert math.isclose(sequence_log_likelihood(obs,pi,A,B),log_likelihood,rel_tol=1e-12,abs_tol=1e-12)

def test_pi_counts_equal_first_gamma_and_xi_row_sums():
 pi,A,B,obs=CASES[0]
 stat=e_step(obs,np.array(pi),np.array(A),np.array(B))
 assert np.allclose(stat['pi_counts'],stat['gamma'][0],atol=1e-14)
 assert np.allclose(stat['xi_sum'].sum(axis=1),stat['gamma'][:-1].sum(axis=0),atol=1e-12)

def test_impossible_sequence_raises_not_silent_zero():
 pi=[1.,0.];A=[[1.,0.],[0.,1.]];B=[[1.,0.],[0.,1.]]
 try:e_step([0,1],np.array(pi),np.array(A),np.array(B))
 except ImpossibleSequence:pass
 else:raise AssertionError('零概率结构下应明确失败')

def test_m_step_zero_rows_keep_previous_and_zeros_stay_zero():
 previous={'pi':[.6,.4],'transition':[[.7,.3],[.4,.6]],'emission':[[.5,.5],[.3,.7]]}
 pi_counts=[0.0,0.0]
 A_counts=np.array([[0.0,0.0],[.4,.6]])
 B_counts=np.array([[.2,0.0],[.0,.5]])
 updated=m_step(pi_counts,A_counts.tolist(),B_counts.tolist(),previous)
 assert updated['pi']==previous['pi']
 assert updated['transition'][0]==previous['transition'][0]
 assert updated['transition'][1]==[.4,.6]
 assert updated['emission'][0][1]==0.0
 assert math.isclose(sum(updated['emission'][0]),1.0)
 assert updated['emission'][1]==[0.0,1.0]

def test_full_round_monotonic_and_preserves_structural_zeros():
 rng=np.random.default_rng(7)
 pi=[.65,.35];A=[[.85,.15],[.4,.6]];B=[[.7,.3],[.2,.8]]
 traces=[rng.integers(0,2,size=6).tolist() for _ in range(12)]
 parameters={'pi':pi,'transition':A,'emission':B}
 first=baum_welch_round(traces,parameters)
 assert first['new_log_likelihood']>=first['previous_log_likelihood']-1e-12
 current=first['parameters']
 for _ in range(8):
  nxt=baum_welch_round(traces,current)
  assert nxt['new_log_likelihood']>=nxt['previous_log_likelihood']-1e-12
  current=nxt['parameters']
  for row in current['transition']+current['emission']+[current['pi']]:
   assert math.isclose(sum(row),1.0,abs_tol=1e-12)
   assert all(v>=0 for v in row)

def test_structural_zero_survives_training():
 rng=np.random.default_rng(11)
 parameters={'pi':[1.,0.],'transition':[[.8,.2],[.0,1.]],'emission':[[.6,.4],[.3,.7]]}
 traces=[rng.integers(0,2,size=5).tolist() for _ in range(10)]
 current=parameters
 for _ in range(6):current=baum_welch_round(traces,current)['parameters']
 assert current['transition'][1][0]==0.0
 assert current['pi'][1]==0.0

def test_batch_counts_are_additive_across_split():
 rng=np.random.default_rng(3)
 parameters={'pi':[.6,.4],'transition':[[.7,.3],[.4,.6]],'emission':[[.5,.5],[.3,.7]]}
 traces=[rng.integers(0,2,size=5).tolist() for _ in range(9)]
 whole=batch_expected_counts(traces,parameters)
 left=batch_expected_counts(traces[:4],parameters);right=batch_expected_counts(traces[4:],parameters)
 assert np.allclose(whole['pi_counts'],np.array(left['pi_counts'])+np.array(right['pi_counts']),atol=1e-12)
 assert np.allclose(whole['transition_counts'],np.array(left['transition_counts'])+np.array(right['transition_counts']),atol=1e-12)
 assert np.allclose(whole['emission_counts'],np.array(left['emission_counts'])+np.array(right['emission_counts']),atol=1e-12)
 assert math.isclose(whole['log_likelihood'],left['log_likelihood']+right['log_likelihood'],abs_tol=1e-12)
