# 16｜实验体系、数据格式、质量门禁与 CI

## 1. 数据目录规范
```text
experiments/<experiment_id>/
  manifest.json
  ruleset.yaml
  metrics/train.jsonl
  metrics/eval_matches.parquet
  events/events.parquet
  snapshots/<episode_id>/
  replays/<episode_id>.jsonl
  checkpoints/{policy,critic}/
  reports/{summary.md,ablation.csv,figures/}
```
记录代码 commit、依赖锁文件、设备/后端、随机种子、地图种子、队伍分配、对手策略 SHA、观察/动作/奖励版本、评测脚本 SHA。

## 2. Manifest 最小字段
```json
{
  "experiment_id": "mappo-baseline-001",
  "git_commit": "<sha>",
  "engine_version": "0.1.0",
  "ruleset_version": "research-v0",
  "observation_version": "entity-v1",
  "action_version": "hybrid-v1",
  "reward_version": "rank-pbrs-v1",
  "algorithm": "mappo",
  "train_seeds": [11, 12, 13, 14, 15],
  "test_seed_manifest": "heldout-v1",
  "opponent_pool_hash": "<sha>",
  "policy_checkpoint_sha": "<sha>",
  "env_steps": 0,
  "physics_steps": 0,
  "counterfactual_branches": 0
}
```

## 3. 对照实验
同一预算比较 IPPO、MAPPO、MAPPO+GRU、MAPPO+PBRS、MAPPO+League、MAPPO+反事实辅助预测、COMA-like。严格固定规则、出生位置集合、对手池、测试种子；不同计算量时同时报告 sample efficiency 和 wall-clock efficiency。独立训练种子至少 5 个作为初始建议，关键结论根据方差做功效分析，不把 5 视为充分保证。

## 4. 赛事评分
主指标：冠军率/胜率、平均名次；次指标：净质量、死亡次数、净分身收益、协作场景成功率、未知队友/对手泛化。采用配对出生位置、交换阵营、bootstrap 置信区间；多重比较控制，报告完整实验而非只报告最佳 seed。可用 Elo/TrueSkill 作辅助排序，但其假设和对手分布需披露。

## 5. CI 四层
- **Unit**：质量计算、碰撞、动作 mask、PBRS、奖励边界、序列化。
- **Property**：质量守恒、实体置换、边界合法、无负质量、时间单调。
- **Integration**：PettingZoo API、死亡/重生、训练 rollout logp、截断 bootstrap。
- **Regression**：黄金快照回放 hash、同 tick 竞争、特定团队协作场景。

## 6. 可复现验收门槛
P0：规则版本明确；所有异常动作行为定义；无负质量；事件总账闭合；回放一致；无 Actor 视野泄漏。
P1：环境 API 校验通过；训练 loss/梯度有限；多个种子基线超过随机/规则对照。
P2：联盟自博弈、反事实归因、零样本协作评测通过完整统计审计。

## 7. 失败报告
每个失败案例包含：scenario、seed、规则版本、state hash、最近 N 步事件、请求动作/实际动作、质量账本差、重现命令和期望/实际结果。不能将“训练跑通”当成算法成功。

## 8. 里程碑交付
S00 完整规则与测量计划；S01 引擎与回放；S02 PettingZoo API；S03 脚本基线；S04 MAPPO；S05 联赛；S06 反事实；S07 消融与论文式报告。每阶段输出验收证据与尚未解决的问题，未运行的阶段标记 NOT IMPLEMENTED。
