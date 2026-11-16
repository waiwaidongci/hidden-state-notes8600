# 燕汐设备隐态研究册

燕汐试验设备组的18位序列分析员把离散观测码与固定HMM假设相对照，再回看独立设备记录。上游是 `yx-fixed-hmm-1-202611.json` 等编码轨迹与已声明参数，下游是负责人批准的工况核对说明。三个具体痛点是单点最高后验被拼成“真实路径”、长序列概率相乘下溢、模型版本与观测来源混淆。

本项目保留固定π/A/B的推断口径（log域前向后向给边际后验，log Viterbi给整段联合MAP；状态名称只是模型假设，不是故障事实或在线告警），并在独立的“页面约束训练”页按多条独立轨迹做带约束的Baum-Welch（EM）：训练只产出新的模型版本，冻结的初始版本、既有推断与已批准核对说明均不被改写。固定seed生成3项研究、3个三态四码模型，每组80条40码轨迹，共240条；推断默认每次选择前20条。

## 启动和角色

```sh
cp .env.example .env
docker compose up --build -d --wait
docker compose exec -T backend python -m scripts.seed_demo
```

|入口|地址|
|---|---|
|浏览器|http://localhost:18237|
|API / Swagger|http://localhost:19237/docs|
|MySQL8|127.0.0.1:17737|

|账号 / 密码|权限与主要页面|典型动作|
|---|---|---|
|admin@yanxi.demo / Yanxi2026!|六页、写入与审批|批准/退回固定模型工况核对|
|analyst@yanxi.demo / Yanxi2026!|六页、导入/版本/轨迹/运行/说明|冻结固定参数、修订观测、排队；审批403|
|viewer@yanxi.demo / Yanxi2026!|六页只读、批准CSV|对照后验和路径、导出；写入403|

`.env.example`含FE/API/DB端口、MYSQL_DATABASE/USER/PASSWORD/ROOT_PASSWORD、JWT_SECRET、SEED_DEMO、RATE_LIMIT_WRITES（默认60）。演示密码和JWT secret仅用于本地，部署须设置自己的值。首次启动自动seed；独立重复执行幂等，不覆盖用户轨迹、版本文字和说明。

Compose恰好frontend/backend/worker/db/redis五服务，各有健康检查。Nginx `/api/`移除前缀后转发FastAPI，前端探针127.0.0.1；backend等DB/Redis，worker再等backend。真实Celery任务保存JSON与数据库结果，backend/worker共享artifacts卷。停止清除本题演示数据：`docker compose down -v --remove-orphans`。

## 一天的序列研究剧本

1. 09:00，许汐登录“隐态总览”，先读固定参数、非故障事实的解释边界。
2. 在“观测与版本”核对 `YX-33A 常规观测序列`、`yx-fixed-hmm-1-202611.json` 和 `YX1-T001` 的40个观测码；字母表A/B/C/D与模型发射列顺序必须一致。
3. 对照 `YX1-固定参数-v1` 的π、转移A、发射B。真实0保持0，不加epsilon；任何超行和容差的输入拒绝。
4. “固定模型推断”默认选前20条，可清空/勾选1–30条。选模型版本后提交；任务冻结profile与所选轨迹，不更新参数。
5. A研究20条/800码，实际总logL约−969.126197943428。首条logL约−49.50643547，线性概率以3.159572e−22等科学记数法展示，不误显示为0。
6. 在“后验与联合路径”逐轨迹看真实状态带、每时刻后验和Viterbi联合logP；单点最高态拼接可能违反零转移，不能直接当联合MAP。
7. 核对 `YX-33C 受限发射序列`：若新观测出现初态不可能发射的C，任务明确失败，不把数值下溢当不可能，也不把0改成小正数。
8. 提交工况核对说明，回看设备记录和码含义。顾临批准/退回；程沅下载批准快照CSV。重复审批409，后续改轨迹不改写既有结果。
9. 在“过程留痕”核对导入、版本冻结、轨迹修订、队列、审批和导出；HTTP响应含X-Request-ID。

## 固定HMM数值口径

π初态分布、A转移、B发射均有限且0≤p≤1；维度2–5状态、2–6观测码。π和每个A/B行的和必须在1±1e−8内，微小偏差才除以该行原始和，保存raw参数、原始行和、delta与是否归一。真实0除后仍0，不设概率地板。符号与状态声明顺序固定；模型版本冻结后不可编辑，探索不同参数须另建版本。

对观测oₜ，`logα₀(s)=logπ(s)+logB(s,o₀)`，`logαₜ(j)=logB(j,oₜ)+logsumexpᵢ(logαₜ₋₁(i)+logA(i,j))`。后向从末步logβ=0递推，序列logL是末步α的logsumexp，边际后验由`exp(logα+logβ−logL)`得到并校正浮点列和。

Viterbi在相同log空间用max替代sum，保留每个状态的最佳前缀路径。相同浮点log分数时选状态索引序列的字典序最小者（比较最早不同一步），边际并列取最小状态index；没有随机平局。有限浮点不承诺无限精度的数学平局辨识。

零概率取−∞；返回JSON的log数组用null明确表示−∞，不是漏测。若某步所有路径均零，抛出“不可能序列”并标整个任务failed，不拼接部分成功结果。若线性概率相乘下溢但logL有限，仍正常完成，线性概率字段为null并单独标记下溢。200步、单步发射1e−4的校验得到logL−1842.0680743952而线性概率下溢。

标准二态三步校验π=.6/.4，A=.7,.3/.4,.6，B=.5,.4,.1/.1,.3,.6，观测0/1/2：全部8路径精确likelihood907/25000，logL−3.316488653735，联合MAP001。

另一个三态27路径校验π=[1,0,0]，A=[[.5,.5,0],[0,0,1],[0,0,1]]，B=[[.5,.3,.2],[.4,.4,.2],[.45,.45,.1]]：likelihood1/40，联合MAP012，边际最高态002却违反A[0,2]=0。真实浏览器导入、冻结、推断、说明与CSV走通了此例。

跨轨迹的logL和是所选记录的描述性总和；将其解释为联合likelihood还需要轨迹间独立。后验和路径都条件于固定模型，未估计参数不确定性、真实故障准确率或在线告警能力。

## 页面约束训练（分批 Baum-Welch）

“页面约束训练”页用多条独立轨迹一起学习初态π、转移A、发射B，不再手填参数。提交时同时冻结初始模型版本与所选轨迹快照（`training_jobs.profile_snapshot/trace_snapshot`），之后修订源记录不影响训练输入；旧推断与已批准核对说明完全不变。

- 期望计数：每条轨迹在log域跑前向后向，得到γ（状态边际）、ξ（相邻状态对期望数）与按观测码累计的发射期望数；批内对各轨迹求和后写入 `training_batches`（π/转移/发射期望计数、批次logL、码数、状态pending/saved），`UNIQUE(job_id,iteration,batch_index)` 保证每个轮次槽位只有一行。
- 零约束：M步只做期望计数行归一。预设为0的位置期望计数恒为0，归一后仍是精确0，不加epsilon或概率地板；某行期望计数总和为0（该行无观测支持）时整行保留上一轮原值，不凭空造分布。
- 逐轮总对数似然与下降不发布：每轮记录轮前总logL（各批次logL的`math.fsum`）与候选参数下重新计算的总logL；仅当候选严格更高（容差1e-12防浮点抖动）才接受新参数，否则保留旧参数并在 `iteration_logs` 标记`not_increase_rejected`。拒绝轮或增益低于设定容差即收敛，最终发布且只发布一个新的 `ModelProfile`（命名含任务id，天然唯一），可直接用于固定模型推断。
- 分批与崩溃恢复：每轮按`batch_size`切批，Celery任务持120秒可续约租约（`lease_owner/lease_expires_at`）。批次行用条件`UPDATE ... WHERE status='pending'`写入并带租约归属校验：进程在批次写入前崩溃，行仍为pending，重启重算该批；写入后崩溃，行已saved，重启跳过、不重算。重复投递在completed状态直接返回；旧执行的租约过期后才允许新进程接管，旧持有者的迟到写入被租约校验拦截，不重复累计、不覆盖新轮。
- 观测在当前模型下出现全零路径时，E步抛“不可能序列”，任务标记failed且不发布半成品模型；不会把数值下溢误当不可能。
- 校验：短序列（2态、3–4步、含结构零）对全部K^T条状态路径穷举，逐项核对γ/ξ/发射期望计数与logL；另用注入崩溃的SQLite测试实测“写入前/写入后中断→租约过期→恢复”，断言已保存批次不重算、最终只产生一个可推断版本，以及API层403/422/409与快照冻结。

## 实体、工程与校验

在原五实体SequenceStudy → ModelProfile → ObservationTrace → InferenceRun → InterpretationNote之外，新增TrainingJob与TrainingBatch两个实体，贯穿ORM/schema/service/routes、前端API/store和七页：隐态总览、观测与版本、固定模型推断、页面约束训练、后验与联合路径、工况核对、过程留痕。训练审计动作`training.queued/training.published/training.failed`自动进入过程留痕。

前端 `src/{api,stores,types,components/common,hooks,pages,router,utils,constants}`；后端 `src/{routes,controllers,services,models,schemas,middlewares,utils,constants,config}`，另scripts/seed_demo.py与migrations/。StateBands、InferenceSummary、ProfileMatrix、RunList多页复用，PosteriorGrid只展示后端后验；useTask/useRoute与format处理交互。Vue的id=app/mount('#app')匹配。

Role/RunStatus/NoteStatus位于 `backend/src/constants/enums.py`，前端对应 `frontend/src/constants/enums.ts`。JWT、审计、RequestID/Error/Redis写限流三个中间件形成横切链路。按令牌摘要或无令牌IP分钟计数，默认60写请求，429含Retry-After；计数不可用503，健康GET不计。审批行锁使draft只审一次，导出取result_snapshot。

每研究1–240轨迹，每条1–200码，单任务1–30条、状态≤5；重复ID、外研究轨迹/版本、未知码、非法矩阵维度、NaN/Inf/bool概率或ID、安全行和以422拒绝。任务提交时同时冻结profile与所选trace，之后修订源记录不影响worker输入或批准快照。

MySQL累计logL及最大归一delta列用DOUBLE(asdecimal=False)，保持Python float。当前MySQL JSON读回与DOUBLE列在累计logL约−969时可差1ULP（约1.14e−13），验收按2ULP检查，完整结果保留实际读回数值。

|层|实现|
|---|---|
|前端|Vue3 / TypeScript / Vite6.4.6，Barlow+Noto Serif SC，石墨/温黄/赤铜状态带|
|后端|Python3.11 / FastAPI / SQLAlchemy / MySQL8|
|队列|Celery5.4 / Kombu5.4.2 / Redis7，worker并发1|
|算法|SciPy logsumexp前后向、手写log Viterbi、真实零约束与归一记录|

本地前端 `cd frontend && npm ci && npm run dev`；`npm run build`执行vue-tsc/Vite。后端安装requirements、配置DATABASE_URL/REDIS_URL/JWT_SECRET后运行uvicorn。ORM create_all初始化，migrations仅初始结构说明，不声称生产迁移链。MIT许可证见LICENSE。

实际验收见批次 `test-artifacts/gb-0033/TESTING.md`，源哈希、测试脚本、JSON、CSV、真实PNG与原始.log均与源码分离。未验证外部真实设备、参数估计/EM、实时处理、生产压力或完整无障碍。令牌/IP限流不是生产抗滥用全链路，换登录令牌可获得新计数。Google字体断网回退本地字体。合成观测没有真实设备故障标签。
