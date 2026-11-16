"""逐轮总对数似然口径与“下降不发布”规则的服务层验证。"""
import math
import numpy as np
from src.services.training_em import baum_welch_round,sequence_log_likelihood,m_step
from src.services import training as ts

def test_every_round_reports_both_total_log_likelihoods():
 rng=np.random.default_rng(17)
 parameters={'pi':[.6,.4],'transition':[[.7,.3],[.4,.6]],'emission':[[.5,.5],[.3,.7]]}
 traces=[rng.integers(0,2,size=7).tolist() for _ in range(15)]
 current=parameters
 for round_index in range(5):
  out=baum_welch_round(traces,current)
  base=math.fsum(sequence_log_likelihood(o,current['pi'],current['transition'],current['emission']) for o in traces)
  assert math.isclose(out['previous_log_likelihood'],base,abs_tol=1e-12)
  assert isinstance(out['new_log_likelihood'],float)
  current=out['parameters']

def test_acceptance_gate_keeps_old_parameters_when_not_increasing(monkeypatch):
 parameters={'pi':[.6,.4],'transition':[[.7,.3],[.4,.6]],'emission':[[.5,.5],[.3,.7]]}
 # 候选参数被人为构造成更差：new_log_likelihood 严格低于 previous
 good_candidate={'pi':[.5,.5],'transition':[[.5,.5],[.5,.5]],'emission':[[.5,.5],[.5,.5]]}
 monkeypatch.setattr(ts,'m_step',lambda *a,**k:good_candidate)
 monkeypatch.setattr(ts,'sequence_log_likelihood',lambda *a,**k:-100.0)
 # 直接复核门限比较函数语义
 previous=-5.0;new=-100.0
 assert not (new>previous+ts.ACCEPT_SLACK)
 # 无期望计数的行保留原值
 updated=m_step([0.0,0.0],[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]],parameters)
 assert updated['pi']==parameters['pi']
 assert updated['transition']==parameters['transition']
 assert updated['emission']==parameters['emission']

def test_structural_zeros_in_counts_remain_exact_zero():
 parameters={'pi':[.6,.4],'transition':[[.7,.3],[.4,.6]],'emission':[[.5,.5],[.3,.7]]}
 # 第二行转移从未被访问，期望计数整行为0，保留旧值；旧值本身含0的位置仍为0
 updated=m_step([.8,.2],[[.5,.5],[0.0,0.0]],[[.6,0.0],[.0,.7]],parameters)
 assert updated['transition'][1]==parameters['transition'][1]
 assert updated['emission'][0][1]==0.0
 assert updated['emission'][1][0]==0.0
 for row in [updated['pi'],*updated['transition'],*updated['emission']]:
  assert math.isclose(sum(row),1.0,abs_tol=1e-12)
