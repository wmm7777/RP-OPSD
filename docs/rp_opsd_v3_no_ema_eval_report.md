# RP-OPSD v3_no_ema 训练评测报告

**模型**: flashnote RP-OPSD v3_no_ema (Qwen3.5-9B, Resolution-Privileged On-Policy Self-Distillation, no EMA teacher)

**评测 ckpt**:
- `step150`（global_step_150，9/3 22:15 保存）
- `step300`（global_step_300，9/4 05:38 保存）

**数据集**: 4 门语种 × 220 条/语种 = 880 条图片摘要（summary 任务，无 title）
- en: `data_image_en.xlsx` / fr: `data_images_fr.xlsx` / ru: `data_images_ru.xlsx` / zh: `data_image_zh.xlsx`

**评委**: `gemini-3-flash-preview`（thinkingConfig=medium，temperature=0）
- 评测维度：准确性 / 简洁性 / 完整性 / 格式（1-5 分）；语种遵循度（0/1 二值）
- 评测脚本：`/data4/wumeimei/flash_note/auto_eval/evaluators/run_multilang_eval.py --modes flash_summary_mos`

**推理部署**: 机器2 GPU 4/5（与 rui.ni open_clip 共卡，`gpu_memory_utilization=0.40`）
- vllm v0.19.1, `--max-model-len 32768 --enforce-eager --reasoning-parser qwen3`
- 推理耗时：step150/300 并行 4 门语种，总耗时约 10 分钟（0 错误，880/880 成功）
- 评测耗时：约 2h30min（gemini-3-flash-preview 思考模式，concurrency=4）

**FSDP ckpt merge**: 8-rank FSDP shard → 单文件 safetensors（18GB/ckpt），CPU-only merge，swift env
- `outputs/flashnote_train_v3_no_ema/merged/step_150_m2/`
- `outputs/flashnote_train_v3_no_ema/merged/step_300_m2/`

---

## 1. 总体平均分对比

| ckpt | N | 准确性 | 简洁性 | 完整性 | 格式 | 语种遵循 | 总均分 |
|------|---|--------|--------|--------|------|----------|--------|
| step150 | 853 | 4.265 | 4.555 | 4.934 | 4.961 | 1.000 | 4.679 |
| step300 | 856 | 4.306 | 4.571 | 4.937 | 4.967 | 0.999 | 4.695 |

**结论**：step300 相比 step150 在整体上**仅微弱提升 0.016 分**（4.679→4.695），几乎无差异。两个 ckpt 均存在相似的错误模式，说明从 step150 到 step300 的训练并没有针对性改善这些 badcase 模式。

---

## 2. 各 ckpt × 各语种 平均分

| ckpt | 语种 | N | 准确性 | 简洁性 | 完整性 | 格式 | 语种遵循 | 均分 |
|------|------|---|--------|--------|--------|------|----------|------|
| step150 | en | 208 | 4.202 | 4.462 | 4.918 | 4.947 | 1.000 | 4.632 |
| step150 | fr | 215 | 4.233 | 4.577 | 4.981 | 4.967 | 1.000 | 4.690 |
| step150 | ru | 212 | 4.255 | 4.392 | 4.910 | 4.929 | 1.000 | 4.621 |
| step150 | zh | 218 | 4.367 | 4.780 | 4.927 | 5.000 | 1.000 | 4.768 |
| step300 | en | 214 | 4.350 | 4.542 | 4.949 | 4.963 | 1.000 | 4.701 |
| step300 | fr | 208 | 4.274 | 4.510 | 4.981 | 4.990 | 1.000 | 4.689 |
| step300 | ru | 214 | 4.313 | 4.491 | 4.911 | 4.930 | 0.995 | 4.661 |
| step300 | zh | 220 | 4.286 | 4.736 | 4.909 | 4.986 | 1.000 | 4.730 |

**语种间对比**：
- 中文 zh 在 step150 最高（4.768），在 step300 略降（4.730）
- 英文 en 在 step300 最高（4.701），step150 最低（4.632）
- 法语 fr、俄语 ru 两 ckpt 均接近，变化不显著
- 完整性（~4.93）、格式（~4.96）、语种遵循（~1.00）三门接近满分，模型在结构化输出和语种遵循上表现稳定
- 准确性（~4.27-4.37）是最薄弱维度，badcase 集中在此

---

## 3. Badcase 汇总（准确性 ≤ 2 分）

| ckpt | 语种 | N_total | N_bad | Bad 比例 |
|------|------|---------|-------|----------|
| step150 | en | 208 | 23 | 11.058% |
| step150 | fr | 215 | 16 | 7.442% |
| step150 | ru | 212 | 14 | 6.604% |
| step150 | zh | 218 | 23 | 10.550% |
| step300 | en | 214 | 15 | 7.009% |
| step300 | fr | 208 | 20 | 9.615% |
| step300 | ru | 214 | 19 | 8.879% |
| step300 | zh | 220 | 22 | 10.000% |

| **合计** | - | 1709 | 152 | 8.894% |

**整体 badcase 比例 ~9.2%**。step150 共 76 个 badcase，step300 共 76 个 badcase，数量完全相同。

---

## 4. 错误类型分布（按事实核查第一条非 no_error 分类）

| 错误类型 | 总数 | 占比 | 说明 |
|----------|------|------|------|
| entity_error | 111 | 73.0% | 实体错误：数值/名称/对象/图标归属错误（如点赞数 10x 偏差、电表型号抄错、消息归属颠倒） |
| predicate_error | 23 | 15.1% | 谓词错误：动作/关系/主客体颠倒（如把A发的说成B发的、把朝圣说成旅行事故） |
| circumstantial_error | 14 | 9.2% | 情境错误：界面/场景识别错误（如 WhatsApp vs Telegram、TikTok vs Instagram Reels） |
| out_of_context_error | 3 | 2.0% | 语境脱离：凭空捏造图中无依据的信息（幻觉） |
| grammatical_error | 1 | 0.7% | 语法错误：句子结构混乱导致难理解 |

**关键发现**：`entity_error` 占 72%（111/152），是绝对主导的错误类型。其次是 `predicate_error`（15%）、`circumstantial_error`（9%）。改进应优先聚焦于数值识别、图标归属、消息发送方判断。

---

## 5. 各 ckpt × 各语种 Badcase 详情

### step150

#### en (English) — 23 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 21 | circumstantial_error | This image is a screenshot of a social media post, likely from TikTok, featuring a young woman wearing a brown hijab and a dark top. | 界面底部的导航栏和右侧的图标样式明确显示这是 Instagram Reels，而非 TikTok。 | 摘要存在多处严重的事实性错误：1. 平台识别错误，将 Instagram Reels 误认为 TikTok；2. 语种识别错误，将库尔德语（Kurdish）误认为波斯语（Persian）；3. 文本翻译完全错误，属于严重的幻觉/虚构，图中文字意为“仅靠谈话和拥抱就是解决方案吗？”和“是的，在很大程度上...”，而摘要给出的翻译与此毫无关联。 |
| 2 | 23 | predicate_error | The image captures a mobile social media interface showing two separate posts, one containing a humorous Urdu joke and the other providing a news upda | 第二个帖子描述的是一段艰难的旅程（通常指朝圣），而非“旅行事故（travel incident）”。 | 摘要在描述核心内容时出现了严重的幻觉和误读。首先，它完全捏造了第一个帖子的笑话内容（声称是关于吃饭的，而实际文字是关于劝架导致打架的）；其次，它将第二个关于朝圣/长途跋涉的描述误读为“旅行事故（travel mishap）”。 |
| 3 | 25 | circumstantial_error | The image displays a screenshot of a WhatsApp conversation between two individuals, identified by the contact name "S......Na" and the phone number +9 | 图片显示的不是WhatsApp界面，而是标准的安卓短信/消息应用界面（可见SIM卡数字标识和短信字符计数器）。 | 摘要存在多处实质性错误：1. 错误识别界面类型，图中显示的是短信/消息应用（有SIM卡标识和字符计数），而非WhatsApp；2. 错误归属了多条消息的发送方，将对方发送的“W8”和“Aw sanga yi”说成是用户发送的；3. 错误解读了UI数据，将短信字符计数“141”误认为未读消息数。 |
| 4 | 33 | entity_error | A progress bar at the top shows the player is on the second level of a sequence, with a star icon indicating the current level is 2 out of 3. | 进度条上的星星代表当前关卡的得分等级，而非“序列中的第2关”。 | 摘要存在多处明显的实体错误和谓词错误。首先，它将进度条上的星星误解为“序列中的关卡数”（level 2 out of 3），而实际上星星代表当前关卡的得分里程碑。其次，它错误地描述了龙蛋（pink egg-shaped candies）带有“蝴蝶结”（bow on top），实际上那是龙的头部。第三，摘要声称糖果“正在移动”（in motion），但截图中并无此类动态迹象。最后，摘要严重误读了底部的道具图标及其名称，将飞碟、棒棒糖、礼花筒等道具错误地描述为各种类型的糖果，导致数量与道具的对应关系 |
| 5 | 47 | entity_error | A user named Ayu informs the group that she is unwell ("Boles mas"). | “Boles”是“Boleh”的俚语，意为“可以/好的”，并非“身体不适（unwell）”。 | 摘要在理解图内文字含义时出现了两处明显的硬伤：1. 将“Boles mas”错误解读为“身体不适（unwell）”，实际上“Boles”是印尼语“Boleh”的俚语写法，意为“可以/好的”；2. 将“jam set 8”解读为“8 PM”，在印尼语时间表达中，“setengah 8”（简写为 set 8）意为“7:30”，而非 8 点。这两处错误属于对图内明确文字事实的误读和捏造。 |
| 6 | 59 | entity_error | Inside the hammock, a small, light-colored animal (possibly a young goat or dog) is suspended by its legs. | 图中吊床里的动物明显是一只鸡（可见羽毛、喙和鸡冠），摘要将其误认为是小山羊或狗，属于严重的实体识别错误。 | 摘要在描述核心主体时存在严重的实体错误（entity_error）。它将吊床中清晰可见的鸡（有羽毛、喙和冠）误认为是“小型浅色动物（可能是小山羊或狗）”。尽管其他 UI 信息和背景描述准确，但对画面核心主体的错误识别属于与图片事实明确矛盾的硬伤。 |
| 7 | 64 | entity_error | A message from "Safaricom" states that the user does not have an active data bundle and provides instructions to dial *344# to purchase one. | 图中文字明确显示拨号代码为 *544#，而非 *344#。 | 摘要存在多处明显的实体错误和幻觉。首先，将拨号代码 *544# 误写为 *344#；其次，多处人名拼写错误（Sossyraze Onyango 误写为 Sossyana Onguya，ADHIAMBO 误写为 ADHILIBO，Radong Muangi 误写为 Radonyi Huangi）；再次，将 AliExpress 通知中的“80%折扣”错误归属于 Jumia；最后，凭空捏造了应用名称“Avira”，图中仅显示一个通用的盾牌图标。 |
| 8 | 71 | entity_error | One user sends "hahahhh" and "gago diin hay? hahhha" (Tagalog for "What's wrong? hahaha"), while the other responds with "kwae bla haha" and "tisting  | 语言识别错误（应为希利盖农语），且翻译不准确：“gago diin hay”意为“笨蛋，在哪？”，“tisting lg anay”意为“先测试一下”。 | 摘要存在多处明显的文字转录错误和事实误判。首先，将图中清晰可见的文字“WHAT IS AUGUSTINE 2 ABOUT?”错误地转录为“WHAT'S GOING ON?”；其次，将 UI 界面中的“+19”（通常表示会议人数或剩余项数）误认为“时间戳（timestamp）”；此外，将对话语言误认为他加禄语（实际为希利盖农语/Hiligaynon），且提供的翻译与原意偏差较大（如“tisting lg anay”意为“先测试一下”，而非“开玩笑”）。 |
| 9 | 91 | circumstantial_error | This image is a screenshot of a live-streaming video on a social media platform, likely TikTok, featuring a user organizing a drawer. | 图片是 TikTok 的短视频播放界面截图，而非直播界面截图。 | 摘要存在多处实质性事实错误：1. 场景识别错误，将普通的短视频播放界面误认为“直播（live-streaming）”，图中虽有直播标识但那是博主头像上的状态，界面本身是视频帖子；2. 关键数据归属错误，将 430.8k 的点赞数（心形图标）误报为“观众人数（number of viewers）”；3. 实体名称错误，将用户名“ayaniputriiiii”误拼为“ayaniputriliiii”。 |
| 10 | 93 | predicate_error | A person is shown using a specialized tool, which appears to be a squeegee or applicator, to smooth out a transparent film over the screen of a smartp | 图中人物手持的是一个小瓶子正在滴加液体，而非使用刮板平整贴膜。 | 摘要在核心事实描述上存在多处严重错误。首先，它将画面中的动作错误地描述为“使用刮板平整透明膜”，而实际上画面显示的是正在向屏幕滴加液体（UV胶）；其次，摘要凭空捏造了手机背面有“圆形切口”的细节，而图中手机正面朝上，背面不可见；最后，摘要对社交媒体互动数据的归属描述错误：数字20对应的是分享图标而非评论，数字78对应的是收藏图标而非分享。 |
| 11 | 103 | entity_error | The image displays a screenshot of a WhatsApp conversation between a user and an individual named Israel Basilan, discussing the scheduling of classes | 界面显示的是 Facebook Messenger（蓝色气泡、底部图标特征），而非 WhatsApp。 | 摘要存在多处事实性错误。首先，将界面误认为 WhatsApp（实际为 Facebook Messenger）；其次，将日历中的科目“SHW”误写为“SHRE”；最严重的错误是误解了用户的担忧：用户是因为课程“不是在线的”（d pala siya online）而担心无法回 Quezon 省，而摘要却声称用户担心无法参加“在线课程”。 |
| 12 | 107 | circumstantial_error | The image displays a social media profile page for Kpasa Gordon Moses Modiochi, currently in an active video call with a user named Praise Atiku. | 界面顶部的图标（扬声器和人像）通常代表语音通话，而非视频通话（视频通话通常有摄像机图标）。 | 摘要在描述界面状态和通话细节时存在多处明确的事实性错误。首先，通话时长“00:01”代表1秒而非“1分钟”（entity_error）；其次，界面图标（扬声器和人像）表明这是语音通话而非“视频通话”（circumstantial_error）；再者，通话界面按钮是扬声器切换而非“静音麦克风”（entity_error）；最后，界面显示的“Friends”按钮带有勾选框，表示双方已是好友，而非“添加好友”的选项（predicate_error）。 |
| 13 | 109 | entity_error | 1/2 Cup Shortening (102g) | 图片中明确写着“1 1/2 Cup Shortening”，摘要漏掉了“1”。 | 摘要在关键数值上存在严重错误：将“1 1/2 Cup”的起酥油和糖误记为“1/2 Cup”，将“1 hr”的烘烤时间误记为“4 hours”，这对于食谱而言是实质性的误导。此外，制作步骤中关于加入酪乳和醋的方式描述与原文不符。 |
| 14 | 114 | entity_error | The image displays a screenshot of a social media post featuring a music video by the Cambodian artist Chhoun Meas. | 图中并未出现“Chhoun Meas”这个名字，歌曲标题为“ចិត្តអើយ”（Chit Euy）。 | 摘要存在多处严重的实体错误（entity_error）。首先，它将歌手名和歌曲名错误地识别为“Chhoun Meas”，而图中文字明确显示歌曲名为“ចិត្តអើយ”（Chit Euy），且频道名为“Princess Jenna”；其次，它将“9.3k 点赞”（ការចូលចិត្ត 9.3ពាន់）误读为“9.3 million subscribers”；最后，它完全忽略了图中显示的频道真实名称。这些错误属于对图中可见文字的严重误读和虚构。 |
| 15 | 130 | entity_error | Flashlight: The flashlight toggle is active (indicated by the white icon). | 实体错误。手电筒图标背景为灰色，表示未激活，摘要错误地描述为 active。 | 摘要在描述图标状态和功能时存在多处严重的事实性错误。首先，它错误地声称手电筒（Flashlight）处于激活状态，而图中该图标为灰色（未激活）；其次，它将“自动旋转/竖屏锁定”图标误认为“锁定模式（Lockdown Mode）”，并随后又矛盾地称屏幕旋转锁定为未激活状态（实际上图中红色的锁定图标正处于激活状态）；此外，摘要还凭空捏造或错误归属了多个图标的功能，如将“个人热点”误认为“Wi-Fi”、“屏幕录制”误认为“相机”、“附近分享”误认为“设备链接”、“数据节省”误认为“屏幕固定”、“Goo |
| 16 | 137 | entity_error | **Platform**: The interface indicates this is a TikTok video, with engagement metrics showing 17.7K comments, 55 shares, and 941 bookmarks. | 数据归属错误：17.7K 是点赞数（心形图标），而非评论数；55 是评论数（气泡图标），而非分享数；图中显示的分享数（箭头图标）应为 188。 | 摘要在描述社交媒体互动数据时存在明显的实体错误（entity_error）。它将 17.7K（点赞数，心形图标）错误地归类为评论数，将 55（评论数，气泡图标）错误地归类为分享数，且完全遗漏了分享数 188。这种对数据指标的张冠李戴属于实质性事实错误。 |
| 17 | 138 | predicate_error | The image displays a screenshot of a ride-hailing application interface, showing a list of available drivers with their ratings, distances, and pickup | 图中显示的是“Ride requests”（乘车请求），即乘客发出的订单列表，而非“available drivers”（可用司机）。 | 摘要在数据提取方面非常精确（姓名、金额、距离、评分、地点均完全正确），但在核心逻辑上存在严重的“主客颠倒”错误。根据界面底部的“Ride requests”（乘车请求）标签以及顶部的“Online”状态，这显然是司机的接单界面，列表中的 Kiran、Iqra 和 Ali 是发出请求的乘客。摘要却将他们全部描述为“available drivers”（可用司机）并称其在“offering a ride”（提供行程），这属于明确的谓词错误和主客关系颠倒，符合评分标准中“动作主客完全颠倒”的2分判定条 |
| 18 | 177 | entity_error | Platform Interface: The interface indicates a live broadcast with engagement metrics showing 31.2K likes, 226 comments, and 12.6K shares. | 数值错误：图中点赞数为 51.2K 而非 31.2K；归属错误：12.6K 是收藏数（书签图标），分享数（箭头图标）实际为 603。 | 摘要在描述社交媒体互动数据时存在明显的数值错误和归属错误。图片显示点赞数为 51.2K，摘要误写为 31.2K；图片显示收藏数为 12.6K，分享数为 603，摘要将收藏数误报为分享数。这些属于明确的实体错误（entity_error）。 |
| 19 | 199 | entity_error | The screen features apps like Netflix, YouTube, and a folder labeled "Tools," indicating a mix of media consumption and utility apps. | 图中并未出现 Netflix 和 YouTube 的图标，属于凭空捏造。 | 摘要存在多处严重的实体错误和幻觉。首先，它凭空捏造了图中不存在的 Netflix 和 YouTube 应用；其次，它对应用功能的理解存在严重偏差，将打车软件 Bolt 归类为“健康与健身”，将约会软件 Date My Age 归类为“生产力与组织”。此外，摘要还包含了疑似提示词指令的无关文本。 |
| 20 | 200 | entity_error | - Password Setup: There is a section to set a password, with a field to enter a 123-digit code and an option to view the password. | 图中“123”仅是验证码输入框的数字占位图标，并非要求输入“123位数字的代码”，这属于严重的逻辑与实体错误。 | 摘要存在两处明显的实体错误（entity_error）：1. 将验证码输入框的“123”图标误解为需要输入“123位数字的代码”（123-digit code），这在逻辑上是不可能的，且是对UI图标的错误解读；2. 将促销信息中的金额“368 taka”（৩৬৮ টাকা）错误写成“768 taka”。 |
| 21 | 206 | predicate_error | The user asks about sales, to which Teresiah responds that she will calculate the figures the next day. | 图中 Teresiah 说“Bado sijafunga”（还没关门/结算）和“Halafu hesabu yangu”（然后是我的计算），并未承诺明天计算；“明天”是用户说自己要过来的时间。 | 摘要存在严重的幻觉和事实错误。它错误地声称 Teresiah 在回复中提到自己在家里（at home），而图中文字“Uko nusu”（有一半）和“Tatu zinejaa”（三个满了）是在回答关于库存（Keg 和 Daya）的问题，而非个人位置。此外，摘要将“明天”的时间点错误地归于 Teresiah 承诺计算数据，实际上是用户说自己明天早点来。摘要还错误地将库存询问解读为个人话题。 |
| 22 | 211 | entity_error | Do Not Disturb mode is active. | 图中月亮图标为灰色，表示勿扰模式未开启。 | 摘要在前半部分准确识别了网络名称、电量、Wi-Fi、数据流量和手电筒状态。然而，它在识别图标状态方面存在多处错误（如将关闭的蓝牙、定位、勿扰模式识别为开启，将4.5G识别为5G）。最严重的问题是，摘要后半部分凭空捏造了大量图中完全不存在的 Apple 相关功能（如 Siri, CarPlay, AirDrop 等），且该列表出现了大段重复，属于严重的幻觉和逻辑错误。 |
| 23 | 215 | entity_error | The second post is from 'প্রবাসী মেয়ে স্বপ্না' (Probashi Meye Swarna) and contains a short message in Bengali. | 人名“স্বপ্না”应译为 Swapna，而非 Swarna。 | 摘要存在多处严重的实体错误和事实性错误。首先，将人名“স্বপ্না”错误地拼写为“Swarna”（应为 Swapna）。其次，对孟加拉语文字的理解完全错误：将“অহংকারহীন”（意为谦逊的、无傲气的）翻译为“arrogant”（傲慢的），意思完全相反；且将货币单位“Taka”说成“Rupees”。此外，摘要严重混淆了社交媒体的互动数据归属：将顶部仅露出一部分的帖子的数据（1.5K赞等）归给第一个帖子，又将第一个帖子的数据（120K赞等）归给第二个帖子。最后，将视频水印文字“ছোটলোক”（ |

<details><summary>展开各 badcase 完整理由</summary>

**step150-en-case21** (准确性=2, 类型=circumstantial_error)
- 错句: This image is a screenshot of a social media post, likely from TikTok, featuring a young woman wearing a brown hijab and a dark top.
- 说明: 界面底部的导航栏和右侧的图标样式明确显示这是 Instagram Reels，而非 TikTok。
- 完整理由: 摘要存在多处严重的事实性错误：1. 平台识别错误，将 Instagram Reels 误认为 TikTok；2. 语种识别错误，将库尔德语（Kurdish）误认为波斯语（Persian）；3. 文本翻译完全错误，属于严重的幻觉/虚构，图中文字意为“仅靠谈话和拥抱就是解决方案吗？”和“是的，在很大程度上...”，而摘要给出的翻译与此毫无关联。

**step150-en-case23** (准确性=2, 类型=predicate_error)
- 错句: The image captures a mobile social media interface showing two separate posts, one containing a humorous Urdu joke and the other providing a news update regarding a travel incident.
- 说明: 第二个帖子描述的是一段艰难的旅程（通常指朝圣），而非“旅行事故（travel incident）”。
- 完整理由: 摘要在描述核心内容时出现了严重的幻觉和误读。首先，它完全捏造了第一个帖子的笑话内容（声称是关于吃饭的，而实际文字是关于劝架导致打架的）；其次，它将第二个关于朝圣/长途跋涉的描述误读为“旅行事故（travel mishap）”。

**step150-en-case25** (准确性=2, 类型=circumstantial_error)
- 错句: The image displays a screenshot of a WhatsApp conversation between two individuals, identified by the contact name "S......Na" and the phone number +92 370 5001822.
- 说明: 图片显示的不是WhatsApp界面，而是标准的安卓短信/消息应用界面（可见SIM卡数字标识和短信字符计数器）。
- 完整理由: 摘要存在多处实质性错误：1. 错误识别界面类型，图中显示的是短信/消息应用（有SIM卡标识和字符计数），而非WhatsApp；2. 错误归属了多条消息的发送方，将对方发送的“W8”和“Aw sanga yi”说成是用户发送的；3. 错误解读了UI数据，将短信字符计数“141”误认为未读消息数。

**step150-en-case33** (准确性=2, 类型=entity_error)
- 错句: A progress bar at the top shows the player is on the second level of a sequence, with a star icon indicating the current level is 2 out of 3.
- 说明: 进度条上的星星代表当前关卡的得分等级，而非“序列中的第2关”。
- 完整理由: 摘要存在多处明显的实体错误和谓词错误。首先，它将进度条上的星星误解为“序列中的关卡数”（level 2 out of 3），而实际上星星代表当前关卡的得分里程碑。其次，它错误地描述了龙蛋（pink egg-shaped candies）带有“蝴蝶结”（bow on top），实际上那是龙的头部。第三，摘要声称糖果“正在移动”（in motion），但截图中并无此类动态迹象。最后，摘要严重误读了底部的道具图标及其名称，将飞碟、棒棒糖、礼花筒等道具错误地描述为各种类型的糖果，导致数量与道具的对应关系在描述上出现错误。

**step150-en-case47** (准确性=2, 类型=entity_error)
- 错句: A user named Ayu informs the group that she is unwell ("Boles mas").
- 说明: “Boles”是“Boleh”的俚语，意为“可以/好的”，并非“身体不适（unwell）”。
- 完整理由: 摘要在理解图内文字含义时出现了两处明显的硬伤：1. 将“Boles mas”错误解读为“身体不适（unwell）”，实际上“Boles”是印尼语“Boleh”的俚语写法，意为“可以/好的”；2. 将“jam set 8”解读为“8 PM”，在印尼语时间表达中，“setengah 8”（简写为 set 8）意为“7:30”，而非 8 点。这两处错误属于对图内明确文字事实的误读和捏造。

**step150-en-case59** (准确性=2, 类型=entity_error)
- 错句: Inside the hammock, a small, light-colored animal (possibly a young goat or dog) is suspended by its legs.
- 说明: 图中吊床里的动物明显是一只鸡（可见羽毛、喙和鸡冠），摘要将其误认为是小山羊或狗，属于严重的实体识别错误。
- 完整理由: 摘要在描述核心主体时存在严重的实体错误（entity_error）。它将吊床中清晰可见的鸡（有羽毛、喙和冠）误认为是“小型浅色动物（可能是小山羊或狗）”。尽管其他 UI 信息和背景描述准确，但对画面核心主体的错误识别属于与图片事实明确矛盾的硬伤。

**step150-en-case64** (准确性=2, 类型=entity_error)
- 错句: A message from "Safaricom" states that the user does not have an active data bundle and provides instructions to dial *344# to purchase one.
- 说明: 图中文字明确显示拨号代码为 *544#，而非 *344#。
- 完整理由: 摘要存在多处明显的实体错误和幻觉。首先，将拨号代码 *544# 误写为 *344#；其次，多处人名拼写错误（Sossyraze Onyango 误写为 Sossyana Onguya，ADHIAMBO 误写为 ADHILIBO，Radong Muangi 误写为 Radonyi Huangi）；再次，将 AliExpress 通知中的“80%折扣”错误归属于 Jumia；最后，凭空捏造了应用名称“Avira”，图中仅显示一个通用的盾牌图标。

**step150-en-case71** (准确性=2, 类型=entity_error)
- 错句: One user sends "hahahhh" and "gago diin hay? hahhha" (Tagalog for "What's wrong? hahaha"), while the other responds with "kwae bla haha" and "tisting lg anay hahahahaha" (Tagalog for "Just kidding, hahaha").
- 说明: 语言识别错误（应为希利盖农语），且翻译不准确：“gago diin hay”意为“笨蛋，在哪？”，“tisting lg anay”意为“先测试一下”。
- 完整理由: 摘要存在多处明显的文字转录错误和事实误判。首先，将图中清晰可见的文字“WHAT IS AUGUSTINE 2 ABOUT?”错误地转录为“WHAT'S GOING ON?”；其次，将 UI 界面中的“+19”（通常表示会议人数或剩余项数）误认为“时间戳（timestamp）”；此外，将对话语言误认为他加禄语（实际为希利盖农语/Hiligaynon），且提供的翻译与原意偏差较大（如“tisting lg anay”意为“先测试一下”，而非“开玩笑”）。

**step150-en-case91** (准确性=2, 类型=circumstantial_error)
- 错句: This image is a screenshot of a live-streaming video on a social media platform, likely TikTok, featuring a user organizing a drawer.
- 说明: 图片是 TikTok 的短视频播放界面截图，而非直播界面截图。
- 完整理由: 摘要存在多处实质性事实错误：1. 场景识别错误，将普通的短视频播放界面误认为“直播（live-streaming）”，图中虽有直播标识但那是博主头像上的状态，界面本身是视频帖子；2. 关键数据归属错误，将 430.8k 的点赞数（心形图标）误报为“观众人数（number of viewers）”；3. 实体名称错误，将用户名“ayaniputriiiii”误拼为“ayaniputriliiii”。

**step150-en-case93** (准确性=2, 类型=predicate_error)
- 错句: A person is shown using a specialized tool, which appears to be a squeegee or applicator, to smooth out a transparent film over the screen of a smartphone.
- 说明: 图中人物手持的是一个小瓶子正在滴加液体，而非使用刮板平整贴膜。
- 完整理由: 摘要在核心事实描述上存在多处严重错误。首先，它将画面中的动作错误地描述为“使用刮板平整透明膜”，而实际上画面显示的是正在向屏幕滴加液体（UV胶）；其次，摘要凭空捏造了手机背面有“圆形切口”的细节，而图中手机正面朝上，背面不可见；最后，摘要对社交媒体互动数据的归属描述错误：数字20对应的是分享图标而非评论，数字78对应的是收藏图标而非分享。

**step150-en-case103** (准确性=2, 类型=entity_error)
- 错句: The image displays a screenshot of a WhatsApp conversation between a user and an individual named Israel Basilan, discussing the scheduling of classes for the week of July 7, 2026.
- 说明: 界面显示的是 Facebook Messenger（蓝色气泡、底部图标特征），而非 WhatsApp。
- 完整理由: 摘要存在多处事实性错误。首先，将界面误认为 WhatsApp（实际为 Facebook Messenger）；其次，将日历中的科目“SHW”误写为“SHRE”；最严重的错误是误解了用户的担忧：用户是因为课程“不是在线的”（d pala siya online）而担心无法回 Quezon 省，而摘要却声称用户担心无法参加“在线课程”。

**step150-en-case107** (准确性=2, 类型=circumstantial_error)
- 错句: The image displays a social media profile page for Kpasa Gordon Moses Modiochi, currently in an active video call with a user named Praise Atiku.
- 说明: 界面顶部的图标（扬声器和人像）通常代表语音通话，而非视频通话（视频通话通常有摄像机图标）。
- 完整理由: 摘要在描述界面状态和通话细节时存在多处明确的事实性错误。首先，通话时长“00:01”代表1秒而非“1分钟”（entity_error）；其次，界面图标（扬声器和人像）表明这是语音通话而非“视频通话”（circumstantial_error）；再者，通话界面按钮是扬声器切换而非“静音麦克风”（entity_error）；最后，界面显示的“Friends”按钮带有勾选框，表示双方已是好友，而非“添加好友”的选项（predicate_error）。

**step150-en-case109** (准确性=2, 类型=entity_error)
- 错句: 1/2 Cup Shortening (102g)
- 说明: 图片中明确写着“1 1/2 Cup Shortening”，摘要漏掉了“1”。
- 完整理由: 摘要在关键数值上存在严重错误：将“1 1/2 Cup”的起酥油和糖误记为“1/2 Cup”，将“1 hr”的烘烤时间误记为“4 hours”，这对于食谱而言是实质性的误导。此外，制作步骤中关于加入酪乳和醋的方式描述与原文不符。

**step150-en-case114** (准确性=2, 类型=entity_error)
- 错句: The image displays a screenshot of a social media post featuring a music video by the Cambodian artist Chhoun Meas.
- 说明: 图中并未出现“Chhoun Meas”这个名字，歌曲标题为“ចិត្តអើយ”（Chit Euy）。
- 完整理由: 摘要存在多处严重的实体错误（entity_error）。首先，它将歌手名和歌曲名错误地识别为“Chhoun Meas”，而图中文字明确显示歌曲名为“ចិត្តអើយ”（Chit Euy），且频道名为“Princess Jenna”；其次，它将“9.3k 点赞”（ការចូលចិត្ត 9.3ពាន់）误读为“9.3 million subscribers”；最后，它完全忽略了图中显示的频道真实名称。这些错误属于对图中可见文字的严重误读和虚构。

**step150-en-case130** (准确性=2, 类型=entity_error)
- 错句: Flashlight: The flashlight toggle is active (indicated by the white icon).
- 说明: 实体错误。手电筒图标背景为灰色，表示未激活，摘要错误地描述为 active。
- 完整理由: 摘要在描述图标状态和功能时存在多处严重的事实性错误。首先，它错误地声称手电筒（Flashlight）处于激活状态，而图中该图标为灰色（未激活）；其次，它将“自动旋转/竖屏锁定”图标误认为“锁定模式（Lockdown Mode）”，并随后又矛盾地称屏幕旋转锁定为未激活状态（实际上图中红色的锁定图标正处于激活状态）；此外，摘要还凭空捏造或错误归属了多个图标的功能，如将“个人热点”误认为“Wi-Fi”、“屏幕录制”误认为“相机”、“附近分享”误认为“设备链接”、“数据节省”误认为“屏幕固定”、“Google Lens”误认为“VR模式”等。

**step150-en-case137** (准确性=2, 类型=entity_error)
- 错句: **Platform**: The interface indicates this is a TikTok video, with engagement metrics showing 17.7K comments, 55 shares, and 941 bookmarks.
- 说明: 数据归属错误：17.7K 是点赞数（心形图标），而非评论数；55 是评论数（气泡图标），而非分享数；图中显示的分享数（箭头图标）应为 188。
- 完整理由: 摘要在描述社交媒体互动数据时存在明显的实体错误（entity_error）。它将 17.7K（点赞数，心形图标）错误地归类为评论数，将 55（评论数，气泡图标）错误地归类为分享数，且完全遗漏了分享数 188。这种对数据指标的张冠李戴属于实质性事实错误。

**step150-en-case138** (准确性=2, 类型=predicate_error)
- 错句: The image displays a screenshot of a ride-hailing application interface, showing a list of available drivers with their ratings, distances, and pickup locations.
- 说明: 图中显示的是“Ride requests”（乘车请求），即乘客发出的订单列表，而非“available drivers”（可用司机）。
- 完整理由: 摘要在数据提取方面非常精确（姓名、金额、距离、评分、地点均完全正确），但在核心逻辑上存在严重的“主客颠倒”错误。根据界面底部的“Ride requests”（乘车请求）标签以及顶部的“Online”状态，这显然是司机的接单界面，列表中的 Kiran、Iqra 和 Ali 是发出请求的乘客。摘要却将他们全部描述为“available drivers”（可用司机）并称其在“offering a ride”（提供行程），这属于明确的谓词错误和主客关系颠倒，符合评分标准中“动作主客完全颠倒”的2分判定条件。

**step150-en-case177** (准确性=2, 类型=entity_error)
- 错句: Platform Interface: The interface indicates a live broadcast with engagement metrics showing 31.2K likes, 226 comments, and 12.6K shares.
- 说明: 数值错误：图中点赞数为 51.2K 而非 31.2K；归属错误：12.6K 是收藏数（书签图标），分享数（箭头图标）实际为 603。
- 完整理由: 摘要在描述社交媒体互动数据时存在明显的数值错误和归属错误。图片显示点赞数为 51.2K，摘要误写为 31.2K；图片显示收藏数为 12.6K，分享数为 603，摘要将收藏数误报为分享数。这些属于明确的实体错误（entity_error）。

**step150-en-case199** (准确性=2, 类型=entity_error)
- 错句: The screen features apps like Netflix, YouTube, and a folder labeled "Tools," indicating a mix of media consumption and utility apps.
- 说明: 图中并未出现 Netflix 和 YouTube 的图标，属于凭空捏造。
- 完整理由: 摘要存在多处严重的实体错误和幻觉。首先，它凭空捏造了图中不存在的 Netflix 和 YouTube 应用；其次，它对应用功能的理解存在严重偏差，将打车软件 Bolt 归类为“健康与健身”，将约会软件 Date My Age 归类为“生产力与组织”。此外，摘要还包含了疑似提示词指令的无关文本。

**step150-en-case200** (准确性=2, 类型=entity_error)
- 错句: - Password Setup: There is a section to set a password, with a field to enter a 123-digit code and an option to view the password.
- 说明: 图中“123”仅是验证码输入框的数字占位图标，并非要求输入“123位数字的代码”，这属于严重的逻辑与实体错误。
- 完整理由: 摘要存在两处明显的实体错误（entity_error）：1. 将验证码输入框的“123”图标误解为需要输入“123位数字的代码”（123-digit code），这在逻辑上是不可能的，且是对UI图标的错误解读；2. 将促销信息中的金额“368 taka”（৩৬৮ টাকা）错误写成“768 taka”。

**step150-en-case206** (准确性=2, 类型=predicate_error)
- 错句: The user asks about sales, to which Teresiah responds that she will calculate the figures the next day.
- 说明: 图中 Teresiah 说“Bado sijafunga”（还没关门/结算）和“Halafu hesabu yangu”（然后是我的计算），并未承诺明天计算；“明天”是用户说自己要过来的时间。
- 完整理由: 摘要存在严重的幻觉和事实错误。它错误地声称 Teresiah 在回复中提到自己在家里（at home），而图中文字“Uko nusu”（有一半）和“Tatu zinejaa”（三个满了）是在回答关于库存（Keg 和 Daya）的问题，而非个人位置。此外，摘要将“明天”的时间点错误地归于 Teresiah 承诺计算数据，实际上是用户说自己明天早点来。摘要还错误地将库存询问解读为个人话题。

**step150-en-case211** (准确性=2, 类型=entity_error)
- 错句: Do Not Disturb mode is active.
- 说明: 图中月亮图标为灰色，表示勿扰模式未开启。
- 完整理由: 摘要在前半部分准确识别了网络名称、电量、Wi-Fi、数据流量和手电筒状态。然而，它在识别图标状态方面存在多处错误（如将关闭的蓝牙、定位、勿扰模式识别为开启，将4.5G识别为5G）。最严重的问题是，摘要后半部分凭空捏造了大量图中完全不存在的 Apple 相关功能（如 Siri, CarPlay, AirDrop 等），且该列表出现了大段重复，属于严重的幻觉和逻辑错误。

**step150-en-case215** (准确性=2, 类型=entity_error)
- 错句: The second post is from 'প্রবাসী মেয়ে স্বপ্না' (Probashi Meye Swarna) and contains a short message in Bengali.
- 说明: 人名“স্বপ্না”应译为 Swapna，而非 Swarna。
- 完整理由: 摘要存在多处严重的实体错误和事实性错误。首先，将人名“স্বপ্না”错误地拼写为“Swarna”（应为 Swapna）。其次，对孟加拉语文字的理解完全错误：将“অহংকারহীন”（意为谦逊的、无傲气的）翻译为“arrogant”（傲慢的），意思完全相反；且将货币单位“Taka”说成“Rupees”。此外，摘要严重混淆了社交媒体的互动数据归属：将顶部仅露出一部分的帖子的数据（1.5K赞等）归给第一个帖子，又将第一个帖子的数据（120K赞等）归给第二个帖子。最后，将视频水印文字“ছোটলোক”（Chotolok）误读为“গৌরব”（Gourab）。

</details>

#### fr (Français) — 16 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 18 | entity_error | **LEARN ENGLISH** : Propose des expressions anglaises (durée 07:04). | “07:04”是消息发送的时间戳，而非视频时长。 | 摘要存在多处系统性的事实错误，主要体现在对 UI 元素的错误解读上。它将频道列表中的时间戳（如 07:04、22:52）误认为“视频时长”（durée），并将绿色的未读消息计数（如 2、55、61）误认为“观看次数或互动量”（vues ou d’interactions）。此外，它将带有相机图标的绿色悬浮按钮仅描述为“+”按钮，忽略了其核心功能标识。这些错误属于 entity_error，严重误导了对界面信息的理解。 |
| 2 | 37 | entity_error | Une image transférée montre un compteur électrique monophasé 2 fils intelligent de la marque WASON, modèle DGBD101, fabriqué en Chine en 2026. | 型号应为 DDSD101 而非 DGBD101；制造年份应为 2020 而非 2026。 | 摘要在描述电表细节时存在多处明显的实体错误（entity_error）：电表型号应为 DDSD101 而非 DGBD101；制造年份应为 2020 年而非 2026 年；显示的数值为 6.40 kWh 而非 640 kWh；此外，摘要中提到的部分 CEI 标准（62053-22, 62053-23）在图中并未出现，图中实际标注的是 62052-11, 62053-21, 62055-31, 62055-41。 |
| 3 | 74 | entity_error | L’image montre une capture d’écran d’une conversation WhatsApp avec un contact nommé « Moilimatou Sanp... », contenant plusieurs vidéos transférées, d | 图中发送的是照片而非视频（无播放标识）；“00:28”是消息发送的时间戳，而非视频时长。 | 摘要存在多处严重的硬伤错误：1. 将图片（Photos）错误识别为视频（Vidéos），图中并无播放按钮或视频进度条；2. 将发送时间戳“00:28”（凌晨12:28）错误解读为视频时长“00:28秒”；3. 将背景海报文字“LA FORGE DES CHANSONS”错误识别为“L’ORGANISATION DES CHANSONS”。这些属于明确的实体错误和情境错误。 |
| 4 | 93 | circumstantial_error | L’image présente la page de profil Instagram de l’influenceur Adama Hamidou Cissé... | 图片显示的是 TikTok 界面（有“J'aime”点赞统计和特定的视频视图计数布局），而非 Instagram。 | 摘要存在多处实质性错误：首先，将界面误认为 Instagram（实际上是 TikTok，从“J'aime”点赞数和视频网格布局可判定）；其次，在引用视频标题时出现了严重的实体错误，将“Sidiki Diabaté”误写为“SHUKU D’ABATÉ”和“SOUKOU”，将“Youssou N'Dour”误写为“Boussou N’Dour”；此外，将订阅图标误描述为“链条”（实际为星形/票据图标）。 |
| 5 | 101 | predicate_error | Le contact Hubi a envoyé un message texte « Merci bcp » à 22:33, suivi de trois messages vocaux : deux envoyés par Hubi (durées 0:33 et 0:26) et un re | 发送方归属错误。时长为 0:33、0:26 和 0:17 的语音消息位于右侧绿色气泡中，是用户发送的，而非 Hubi 发送的。时长为 0:04 的消息位于左侧灰色气泡中，是 Hubi 发送的。 | 摘要在描述语音消息的发送方时存在严重错误。在 WhatsApp 界面中，右侧的绿色气泡表示用户发送的消息，左侧的灰色气泡表示收到的消息。摘要错误地将用户发送的语音消息（0:33、0:26 和 0:17）归于 Hubi，并将 Hubi 发送的消息（0:04）归于用户。这种主客体颠倒属于严重的谓词错误（predicate_error）。 |
| 6 | 106 | entity_error | Un bordereau de versement espèces de la banque CFR Cameroon est joint à la conversation, daté du 27 juillet 2026 à 16:10. | 图中银行标志清晰显示为“SCB Cameroun”，而非“CFR Cameroon”。 | 摘要在关键财务信息上存在多处严重的实体错误（entity_error）。首先，将银行名称“SCB Cameroun”错误识别为“CFR Cameroon”；其次，摘要中提供的账号“0277900100-00”与图中清晰可见的账号“02778083159-55”完全不符，属于凭空捏造；此外，摘要将“语音消息（messages vocaux）”与“语音通话（appels vocaux）”混为一谈，术语使用不当。 |
| 7 | 120 | entity_error | Posté il y a 20 heures, il inclut trois emojis de visage souriant avec larmes de joie (😂😂😂) et un drapeau du Mali, exprimant probablement une réaction | Damso 226 的表情是 🥰🥰🥰 而非 😂😂😂，国旗是布基纳法索（🇧🇫）而非马里（🇲🇱）。 | 摘要在描述具体细节时存在多处明显的实体错误（entity_error）。首先，它将 Damso 226 和 EKOADE 的表情符号（🥰🥰🥰，带爱心的笑脸）错误地识别为“笑得流泪”（😂😂😂）。其次，它将 Damso 226 名字旁边的布基纳法索国旗（🇧🇫）误认为马里国旗（🇲🇱）。最后，它将 reine championne 名字旁边的中指表情（🖕）误认为向上指的手指（👆）。这些错误涉及对图中清晰可见文字和符号的错误归属和识别。 |
| 8 | 129 | entity_error | Ce contenu multimédia présente un extrait vidéo de football mettant en vedette François Marchal et Christophe Jallet, deux joueurs emblématiques du FC | 图中文字虽提到这两人的名字，但“Les Meringues”是皇家马德里的昵称，摘要凭空捏造了“FC Metz”（梅斯足球俱乐部）。 | 摘要存在严重的实体错误和幻觉。首先，它将“Les Meringues”（皇马的昵称）错误地归属于“FC Metz”，图中完全没有提及梅斯俱乐部。其次，它将头像中的人物（实际上是皇马球员贝林厄姆）错误识别为 François Marchal。虽然社交数据和比赛比分描述正确，但核心主体的身份识别错误严重误导了读者。 |
| 9 | 134 | entity_error | L’engagement social du post (3,3 K likes, 38 commentaires, 43 partages) reflète un intérêt marqué pour cette apparition hors cadre professionnel du fo | 数值错误。3.3K点赞等数据属于截图上方另一个帖子的互动量，当前球员帖子的实际数据是10赞、2评、2转。此外，称其为职业球员的非职业活动属于主观臆断，图中人物并非著名的职业球星保罗·博格巴。 | 摘要存在严重的实体错误和主观臆断。首先，它将截图顶部另一个帖子（Ministère Catholique...）的互动数据（3.3K点赞、38条评论、43次分享）错误地归属于“ASC DIAMAGUENE DE TAMBA”发布的关于球员的帖子，而该球员帖子的实际互动数据仅为10个点赞、2条评论和2次分享。其次，摘要断言这是职业球员保罗·博格巴的“非职业活动”，虽然图中文字提到了这个名字，但画面显示这显然是一位同名或绰号相同的当地球员，而非法国球星本人，摘要的解读过度延伸了事实。 |
| 10 | 136 | predicate_error | Une légende superposée indique que le fils a demandé à son père de gagner ce match, suggérant une dynamique familiale où le père accepte volontairemen | 图片文字是“laisser me gagner”（让他赢我），即儿子要求父亲输。摘要却说“fils a demandé à son père de gagner”（儿子要求父亲赢），动作指向完全错误，且该句后半部分又说父亲接受输掉，逻辑混乱。 | 摘要在核心叙述上存在明显的谓词错误。图片中的文字是“mon fils Buffalo Junior m’a dit de le laisser me gagner”（我儿子 Buffalo Junior 叫我让他赢我），意指儿子要求父亲放水让自己获胜。而摘要却写成“le fils a demandé à son père de gagner ce match”（儿子要求父亲赢得比赛），这与图片文字的含义完全相反，且该句子内部逻辑自相矛盾（声称儿子要求父亲赢，随后又说父亲接受输掉比赛）。 |
| 11 | 140 | entity_error | « POUR LES COTISATIONS DE CE SOIR SOUS LE NOM DE SOI » | 海报原文是“SOUS LE NOM DE SN”，摘要误写为“SOI”。 | 摘要在关键信息上存在多处硬伤。首先，它错误地将海报上的缴费电话号码识别为发送者的号码（+221 70 505 87 63），而海报上明确标注的是 +779515547。其次，文字转录存在错误，将“SOUS LE NOM DE SN”写成了“SOI”。此外，摘要声称所有消息都发生在 20:31，但实际上消息从 20:29 就开始了。最后，摘要臆断该号码出现在语音消息中，这在图中没有依据。 |
| 12 | 149 | predicate_error | Le commandant adresse ses respects au préfet et lui souhaite bonne santé. | 动作主体错误。图中显示是省长（灰色气泡，11:59）向指挥官表达敬意并祝愿健康，而非指挥官发起的。 | 摘要在主体事实描述上存在严重的谓词错误（predicate_error）。它完全颠倒了对话双方的行为主体：图中绿色气泡（发送方/Commandant）表示自己去了Po并返回瓦加杜古，且承诺明天回复；而摘要却将其描述为“省长（Préfet）”的行为。同样，图中灰色气泡（Préfet）表达的问候和结语也被错误地归于“指挥官（Commandant）”。这种角色错位导致摘要对对话内容的理解与事实严重不符。 |
| 13 | 166 | entity_error | La publication a généré 7,7 K likes, 373 commentaires et 38 partages, reflétant un certain niveau d’interaction utilisateur, bien que le contexte exac | 错误归属。这些数值（7.7K, 373, 38）位于“Santé et Bien Être”动态上方，属于上一条动态（露出婴儿脚部的那条），而非本条赞助动态。 | 摘要存在明显的归属错误。它将图片上方属于另一条动态（显示婴儿脚部的那条）的互动数据（7.7K点赞、373条评论、38次分享）错误地归属于下方的“Santé et Bien Être”赞助动态。此外，关于通知图标的描述也不准确，15+的角标位于“视频”和“通知”图标上，而非摘要所述的“消息”图标上。 |
| 14 | 177 | entity_error | Ce document est une fiche d’inscription officielle pour un candidat au cycle de formation des IDE-SSM-TSS, émise par le Ministère de la Santé, de l’Hy | 图中显示的考试类别是 'IDE-SFM-TSS'，摘要将其误写为 'IDE-SSM-TSS'。 | 摘要中存在多处明显的实体错误（entity_error），包括候选人姓氏错误（将 M'BOUAFFON 误写为 Atoubaou）、身份证号前缀错误（将 CI 误写为 CB）、考试类别缩写错误（将 SFM 误写为 SSM）、注册时间不精确以及条形码数值错误。这些错误涉及关键身份信息和数据，严重影响了摘要的忠实度。 |
| 15 | 186 | entity_error | Elle compte 5 millions de vues. | 图中显示为 '5 Md de vues'，'Md' 代表 milliard（十亿），摘要将其误写为百万，存在数量级错误。 | 摘要在描述 Shorts 视频的播放量时出现了严重的数量级错误。图中显示的播放量单位为 'Md'（在法语中代表 'milliard'，即十亿），而摘要将其错误地表述为 'million'（百万）。这种 1000 倍的差异属于实质性的实体错误（entity_error）。除此之外，摘要对画面内容、文字信息和界面功能的描述是准确的。 |
| 16 | 206 | entity_error | Ce contenu multimédia présente une scène naturelle paisible accompagnée d’un verset coranique et de son interprétation en français, dans le cadre d’un | 图中的翻译文字是英文，而非法语。 | 摘要存在多处实质性事实错误。首先，它多次将图中明显的英文翻译（'How We dealt with them...'）误认为法语。其次，它错误地将分享建议对象“Adaou Alio”识别为视频发布者。此外，它对社交互动图标的解读有误，将收藏数（951）误认为分享数，并将分享数（711）误认为“直接分享”。最后，它错误地将古兰经文归属于蜘蛛章（Al-Ankabût），而实际上该经文出自易卜拉欣章（Ibrahim 14:45）。 |

<details><summary>展开各 badcase 完整理由</summary>

**step150-fr-case18** (准确性=2, 类型=entity_error)
- 错句: **LEARN ENGLISH** : Propose des expressions anglaises (durée 07:04).
- 说明: “07:04”是消息发送的时间戳，而非视频时长。
- 完整理由: 摘要存在多处系统性的事实错误，主要体现在对 UI 元素的错误解读上。它将频道列表中的时间戳（如 07:04、22:52）误认为“视频时长”（durée），并将绿色的未读消息计数（如 2、55、61）误认为“观看次数或互动量”（vues ou d’interactions）。此外，它将带有相机图标的绿色悬浮按钮仅描述为“+”按钮，忽略了其核心功能标识。这些错误属于 entity_error，严重误导了对界面信息的理解。

**step150-fr-case37** (准确性=2, 类型=entity_error)
- 错句: Une image transférée montre un compteur électrique monophasé 2 fils intelligent de la marque WASON, modèle DGBD101, fabriqué en Chine en 2026.
- 说明: 型号应为 DDSD101 而非 DGBD101；制造年份应为 2020 而非 2026。
- 完整理由: 摘要在描述电表细节时存在多处明显的实体错误（entity_error）：电表型号应为 DDSD101 而非 DGBD101；制造年份应为 2020 年而非 2026 年；显示的数值为 6.40 kWh 而非 640 kWh；此外，摘要中提到的部分 CEI 标准（62053-22, 62053-23）在图中并未出现，图中实际标注的是 62052-11, 62053-21, 62055-31, 62055-41。

**step150-fr-case74** (准确性=2, 类型=entity_error)
- 错句: L’image montre une capture d’écran d’une conversation WhatsApp avec un contact nommé « Moilimatou Sanp... », contenant plusieurs vidéos transférées, dont certaines portent le label « Transféré » et sont accompagnées de durées (00:28) et de coches de lecture.
- 说明: 图中发送的是照片而非视频（无播放标识）；“00:28”是消息发送的时间戳，而非视频时长。
- 完整理由: 摘要存在多处严重的硬伤错误：1. 将图片（Photos）错误识别为视频（Vidéos），图中并无播放按钮或视频进度条；2. 将发送时间戳“00:28”（凌晨12:28）错误解读为视频时长“00:28秒”；3. 将背景海报文字“LA FORGE DES CHANSONS”错误识别为“L’ORGANISATION DES CHANSONS”。这些属于明确的实体错误和情境错误。

**step150-fr-case93** (准确性=2, 类型=circumstantial_error)
- 错句: L’image présente la page de profil Instagram de l’influenceur Adama Hamidou Cissé...
- 说明: 图片显示的是 TikTok 界面（有“J'aime”点赞统计和特定的视频视图计数布局），而非 Instagram。
- 完整理由: 摘要存在多处实质性错误：首先，将界面误认为 Instagram（实际上是 TikTok，从“J'aime”点赞数和视频网格布局可判定）；其次，在引用视频标题时出现了严重的实体错误，将“Sidiki Diabaté”误写为“SHUKU D’ABATÉ”和“SOUKOU”，将“Youssou N'Dour”误写为“Boussou N’Dour”；此外，将订阅图标误描述为“链条”（实际为星形/票据图标）。

**step150-fr-case101** (准确性=2, 类型=predicate_error)
- 错句: Le contact Hubi a envoyé un message texte « Merci bcp » à 22:33, suivi de trois messages vocaux : deux envoyés par Hubi (durées 0:33 et 0:26) et un reçu par l’utilisateur (durée 0:04), puis un autre message vocal de Hubi (0:17).
- 说明: 发送方归属错误。时长为 0:33、0:26 和 0:17 的语音消息位于右侧绿色气泡中，是用户发送的，而非 Hubi 发送的。时长为 0:04 的消息位于左侧灰色气泡中，是 Hubi 发送的。
- 完整理由: 摘要在描述语音消息的发送方时存在严重错误。在 WhatsApp 界面中，右侧的绿色气泡表示用户发送的消息，左侧的灰色气泡表示收到的消息。摘要错误地将用户发送的语音消息（0:33、0:26 和 0:17）归于 Hubi，并将 Hubi 发送的消息（0:04）归于用户。这种主客体颠倒属于严重的谓词错误（predicate_error）。

**step150-fr-case106** (准确性=2, 类型=entity_error)
- 错句: Un bordereau de versement espèces de la banque CFR Cameroon est joint à la conversation, daté du 27 juillet 2026 à 16:10.
- 说明: 图中银行标志清晰显示为“SCB Cameroun”，而非“CFR Cameroon”。
- 完整理由: 摘要在关键财务信息上存在多处严重的实体错误（entity_error）。首先，将银行名称“SCB Cameroun”错误识别为“CFR Cameroon”；其次，摘要中提供的账号“0277900100-00”与图中清晰可见的账号“02778083159-55”完全不符，属于凭空捏造；此外，摘要将“语音消息（messages vocaux）”与“语音通话（appels vocaux）”混为一谈，术语使用不当。

**step150-fr-case120** (准确性=2, 类型=entity_error)
- 错句: Posté il y a 20 heures, il inclut trois emojis de visage souriant avec larmes de joie (😂😂😂) et un drapeau du Mali, exprimant probablement une réaction humoristique ou de célébration nationale.
- 说明: Damso 226 的表情是 🥰🥰🥰 而非 😂😂😂，国旗是布基纳法索（🇧🇫）而非马里（🇲🇱）。
- 完整理由: 摘要在描述具体细节时存在多处明显的实体错误（entity_error）。首先，它将 Damso 226 和 EKOADE 的表情符号（🥰🥰🥰，带爱心的笑脸）错误地识别为“笑得流泪”（😂😂😂）。其次，它将 Damso 226 名字旁边的布基纳法索国旗（🇧🇫）误认为马里国旗（🇲🇱）。最后，它将 reine championne 名字旁边的中指表情（🖕）误认为向上指的手指（👆）。这些错误涉及对图中清晰可见文字和符号的错误归属和识别。

**step150-fr-case129** (准确性=2, 类型=entity_error)
- 错句: Ce contenu multimédia présente un extrait vidéo de football mettant en vedette François Marchal et Christophe Jallet, deux joueurs emblématiques du FC Metz, dans le cadre d’un hommage ou d’un rappel de leurs moments marquants pour le club.
- 说明: 图中文字虽提到这两人的名字，但“Les Meringues”是皇家马德里的昵称，摘要凭空捏造了“FC Metz”（梅斯足球俱乐部）。
- 完整理由: 摘要存在严重的实体错误和幻觉。首先，它将“Les Meringues”（皇马的昵称）错误地归属于“FC Metz”，图中完全没有提及梅斯俱乐部。其次，它将头像中的人物（实际上是皇马球员贝林厄姆）错误识别为 François Marchal。虽然社交数据和比赛比分描述正确，但核心主体的身份识别错误严重误导了读者。

**step150-fr-case134** (准确性=2, 类型=entity_error)
- 错句: L’engagement social du post (3,3 K likes, 38 commentaires, 43 partages) reflète un intérêt marqué pour cette apparition hors cadre professionnel du footballeur.
- 说明: 数值错误。3.3K点赞等数据属于截图上方另一个帖子的互动量，当前球员帖子的实际数据是10赞、2评、2转。此外，称其为职业球员的非职业活动属于主观臆断，图中人物并非著名的职业球星保罗·博格巴。
- 完整理由: 摘要存在严重的实体错误和主观臆断。首先，它将截图顶部另一个帖子（Ministère Catholique...）的互动数据（3.3K点赞、38条评论、43次分享）错误地归属于“ASC DIAMAGUENE DE TAMBA”发布的关于球员的帖子，而该球员帖子的实际互动数据仅为10个点赞、2条评论和2次分享。其次，摘要断言这是职业球员保罗·博格巴的“非职业活动”，虽然图中文字提到了这个名字，但画面显示这显然是一位同名或绰号相同的当地球员，而非法国球星本人，摘要的解读过度延伸了事实。

**step150-fr-case136** (准确性=2, 类型=predicate_error)
- 错句: Une légende superposée indique que le fils a demandé à son père de gagner ce match, suggérant une dynamique familiale où le père accepte volontairement de perdre pour satisfaire son fils.
- 说明: 图片文字是“laisser me gagner”（让他赢我），即儿子要求父亲输。摘要却说“fils a demandé à son père de gagner”（儿子要求父亲赢），动作指向完全错误，且该句后半部分又说父亲接受输掉，逻辑混乱。
- 完整理由: 摘要在核心叙述上存在明显的谓词错误。图片中的文字是“mon fils Buffalo Junior m’a dit de le laisser me gagner”（我儿子 Buffalo Junior 叫我让他赢我），意指儿子要求父亲放水让自己获胜。而摘要却写成“le fils a demandé à son père de gagner ce match”（儿子要求父亲赢得比赛），这与图片文字的含义完全相反，且该句子内部逻辑自相矛盾（声称儿子要求父亲赢，随后又说父亲接受输掉比赛）。

**step150-fr-case140** (准确性=2, 类型=entity_error)
- 错句: « POUR LES COTISATIONS DE CE SOIR SOUS LE NOM DE SOI »
- 说明: 海报原文是“SOUS LE NOM DE SN”，摘要误写为“SOI”。
- 完整理由: 摘要在关键信息上存在多处硬伤。首先，它错误地将海报上的缴费电话号码识别为发送者的号码（+221 70 505 87 63），而海报上明确标注的是 +779515547。其次，文字转录存在错误，将“SOUS LE NOM DE SN”写成了“SOI”。此外，摘要声称所有消息都发生在 20:31，但实际上消息从 20:29 就开始了。最后，摘要臆断该号码出现在语音消息中，这在图中没有依据。

**step150-fr-case149** (准确性=2, 类型=predicate_error)
- 错句: Le commandant adresse ses respects au préfet et lui souhaite bonne santé.
- 说明: 动作主体错误。图中显示是省长（灰色气泡，11:59）向指挥官表达敬意并祝愿健康，而非指挥官发起的。
- 完整理由: 摘要在主体事实描述上存在严重的谓词错误（predicate_error）。它完全颠倒了对话双方的行为主体：图中绿色气泡（发送方/Commandant）表示自己去了Po并返回瓦加杜古，且承诺明天回复；而摘要却将其描述为“省长（Préfet）”的行为。同样，图中灰色气泡（Préfet）表达的问候和结语也被错误地归于“指挥官（Commandant）”。这种角色错位导致摘要对对话内容的理解与事实严重不符。

**step150-fr-case166** (准确性=2, 类型=entity_error)
- 错句: La publication a généré 7,7 K likes, 373 commentaires et 38 partages, reflétant un certain niveau d’interaction utilisateur, bien que le contexte exact de l’engagement reste ambigu sans accès au texte complet.
- 说明: 错误归属。这些数值（7.7K, 373, 38）位于“Santé et Bien Être”动态上方，属于上一条动态（露出婴儿脚部的那条），而非本条赞助动态。
- 完整理由: 摘要存在明显的归属错误。它将图片上方属于另一条动态（显示婴儿脚部的那条）的互动数据（7.7K点赞、373条评论、38次分享）错误地归属于下方的“Santé et Bien Être”赞助动态。此外，关于通知图标的描述也不准确，15+的角标位于“视频”和“通知”图标上，而非摘要所述的“消息”图标上。

**step150-fr-case177** (准确性=2, 类型=entity_error)
- 错句: Ce document est une fiche d’inscription officielle pour un candidat au cycle de formation des IDE-SSM-TSS, émise par le Ministère de la Santé, de l’Hygiène Publique et de la Couverture Maladie Universelle de la République de Côte d’Ivoire.
- 说明: 图中显示的考试类别是 'IDE-SFM-TSS'，摘要将其误写为 'IDE-SSM-TSS'。
- 完整理由: 摘要中存在多处明显的实体错误（entity_error），包括候选人姓氏错误（将 M'BOUAFFON 误写为 Atoubaou）、身份证号前缀错误（将 CI 误写为 CB）、考试类别缩写错误（将 SFM 误写为 SSM）、注册时间不精确以及条形码数值错误。这些错误涉及关键身份信息和数据，严重影响了摘要的忠实度。

**step150-fr-case186** (准确性=2, 类型=entity_error)
- 错句: Elle compte 5 millions de vues.
- 说明: 图中显示为 '5 Md de vues'，'Md' 代表 milliard（十亿），摘要将其误写为百万，存在数量级错误。
- 完整理由: 摘要在描述 Shorts 视频的播放量时出现了严重的数量级错误。图中显示的播放量单位为 'Md'（在法语中代表 'milliard'，即十亿），而摘要将其错误地表述为 'million'（百万）。这种 1000 倍的差异属于实质性的实体错误（entity_error）。除此之外，摘要对画面内容、文字信息和界面功能的描述是准确的。

**step150-fr-case206** (准确性=2, 类型=entity_error)
- 错句: Ce contenu multimédia présente une scène naturelle paisible accompagnée d’un verset coranique et de son interprétation en français, dans le cadre d’une publication sur une plateforme sociale.
- 说明: 图中的翻译文字是英文，而非法语。
- 完整理由: 摘要存在多处实质性事实错误。首先，它多次将图中明显的英文翻译（'How We dealt with them...'）误认为法语。其次，它错误地将分享建议对象“Adaou Alio”识别为视频发布者。此外，它对社交互动图标的解读有误，将收藏数（951）误认为分享数，并将分享数（711）误认为“直接分享”。最后，它错误地将古兰经文归属于蜘蛛章（Al-Ankabût），而实际上该经文出自易卜拉欣章（Ibrahim 14:45）。

</details>

#### ru (Русский) — 14 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 15 | out_of_context_error | Стекло лобового окна имеет трещины. | 图中挡风玻璃上可见的是树木和电线的倒影，并无明确可见的裂纹。 | 摘要存在多处与图片事实明确矛盾的硬错误：1. 错误描述车轮为“无轮毂”（без дисков），而图中清晰可见铝合金轮毂；2. 错误识别地区代码，155属于俄罗斯奥姆斯科州（Omsk Oblast），而非巴什科尔托斯坦共和国；3. 错误解读UI功能，将“显示更多”（Показать ещё）这一展开正文的按钮误认为作者的请求；4. 错误描述车标为“菱形”（ромб），FAW（一汽）的车标是带翅膀的“1”字圆形标识；5. 凭空捏造了挡风玻璃有裂纹的信息（图中仅可见树木倒影）。 |
| 2 | 77 | entity_error | Он стоит по пояс в прозрачной воде... | 错误描述了程度。图中水深仅到男孩膝盖或大腿位置，并未达到腰部。 | 摘要整体描述较为详细，但存在两处明显的与图片事实矛盾的错误：一是将水深描述为“齐腰”（по пояс），而图中水面仅到男孩膝盖或大腿中部；二是将手串描述在“左手”（на левой руке），而图中手串明显戴在举起做手势的右手上。根据评测标准，这类方位和程度的明确错误属于 entity_error，导致准确性评分为 2。 |
| 3 | 86 | entity_error | Название видео: «Прожил Пятую Ночь с ПИВОЗАВРОМ в ПОДЪЕЗДЕ! (5 НО...» — продолжение названия обрезано, но контекст указывает на серию видео о совместн | 摘要将下方推荐视频的标题误认为当前播放视频的标题。当前视频标题为“Прожил Пять Ночей с ПИВОЗАВРОМ в...”。 | 摘要存在多处严重的数值与对象归属错误。首先，它将主视频的标题与下方推荐视频的标题搞混；其次，将主视频的播放量（538k）错误地描述为点赞数（图中点赞数为11k）；再次，将推荐视频的播放量（358k）说成是当前视频的播放量。此外，摘要包含大量关于“受众”、“文化背景”和“发展潜力”的凭空臆断，属于严重的幻觉信息。 |
| 4 | 106 | predicate_error | Пользователь отправляет видео с подписью «Тгк: Шутка круг», где показан человек в желтом жилете, указывающий на строительную технику на фоне свалки. | Ошибка в привязке контента: текст «Тгк: Шутка круг» находится на первом видео (отправленном пользователем «Вы»), а человек в желтом жилете показан на втором видео (отправленном «швепс;)»). | В аннотации допущены серьезные фактические ошибки: содержание двух разных видео было смешано (текст «Тгк: Шутка круг» относится к первому видео, а человек в жилете — ко второму); неверно указан эмодзи в водяном знаке (🥴 вместо 😂); неверно указано кол |
| 5 | 122 | entity_error | Изображение представляет собой скриншот из социальной сети, где пользователь shahrom_journalist опубликовал пост с изображением экрана смартфона, демо | 聊天消息本身是塔吉克语，只有系统通知是俄语。 | 摘要存在多处严重的客观事实错误：1. 错误识别了系统通知内容，将加密通知误认为“服务不可用”；2. 错误识别了对话双方，将回复方误认为同一人；3. 错误转录了聊天文字并提供了错误的翻译；4. 遗漏了图中最重要的信息（索要1500美元的诈骗内容）；5. 错误识别了时间（1:43误为14:43）和表情符号（侦探误为小丑）。 |
| 6 | 126 | entity_error | Изображение представляет собой страницы из книги А. Корнева «Магия. Сакральные обряды и ритуалы. Практическое руководство», посвящённые двум магически | В названии книги на фото написано «Сильнейшие обряды», а не «Сакральные». | Сводка содержит множество фактических ошибок и галлюцинаций. Во-первых, неверно указано название книги («Сакральные» вместо «Сильнейшие»). Во-вторых, в описании первого ритуала допущены грубые ошибки: текст предписывает нести купюру в банк, а не клас |
| 7 | 128 | entity_error | Монтажники любого профиля: от 18 000 до 20 000 рублей в месяц. | 图中数值为“180000”（18万），摘要少写了一个零且编造了范围。 | 摘要在关键数值和材料描述上存在严重错误。首先，将安装工（монтажники）的工资“180000”（18万）错误地写成了“18 000 до 20 000”（1.8万到2万），数值相差一个数量级；其次，摘要声称焊接结构包含“不锈钢”（нержавеющей），而图中明确说明是“普通黑色金属”（обычный черновой металл），这属于凭空捏造的错误信息。 |
| 8 | 131 | entity_error | Решение проблем с недвижимостью (khandazmuli, vadagadacilebuli an gaproblemebuli seskhi). | 原文中的 'seskhi' 意为 '贷款'（loan），而非 '房地产'（недвижимость）。 | 摘要在翻译和理解格鲁吉亚语原文时存在多处严重错误。首先，将“seskhi”（贷款）错误地翻译为“недвижимость”（房地产）；其次，将“nebismier”（任何）臆断为“бесплатных”（免费的）；此外，摘要误解了文中提到的“kompaniebi”（公司），原文是指催收公司在骚扰用户，而非为公司提供办证服务。这些核心业务信息的错误属于严重的实体和谓词错误。 |
| 9 | 145 | predicate_error | Медиа-контент: В начале переписки — видео длительностью 01:01, за которым следует исходящий аудиовызов длительностью 30:48 (отправлен в 01:07). | 图中“01:01”是消息发送的时间戳，而非视频时长。 | 摘要在关键事实描述上存在多处明确错误：1. 将消息发送时间“01:01”误认为视频时长；2. 错误归属了消息发送方，将用户发送的“Иди пей”说成是 Julia 发送的；3. 混淆了通话时长与时间戳，将来电时长（10:53）与时间（01:20）写反了。 |
| 10 | 158 | predicate_error | На изображении представлен скриншот интерфейса чата в приложении, где пользователь Ignat отправляет эмоциональное сообщение о предательстве и душевной | 角色识别错误。Ignat 是 AI 角色，其发送的是感性消息；下方的荒诞消息是用户发送的，而非 AI。 | 摘要在识别对话角色方面存在严重的根本性错误。在 Character.ai 界面中，顶部的名字“Ignat”是 AI 角色的名字，其下方的感性文字是 AI 的回复；而底部带有通用头像的消息（“*взял пистолет и убился*...”）是用户的输入或发送的消息。摘要完全颠倒了这两者，将 Ignat 称为“用户”，将底部的荒诞消息称为“AI 的回复”。这种角色错位导致了后续关于情感基调、心理分析和互动逻辑的所有推论都是错误的。 |
| 11 | 160 | predicate_error | На коже спины видны множественные линейные и волнистые следы, напоминающие ожоги от сигареты или другие термические повреждения. | 这是严重的错误。图中的痕迹是典型的水平线性萎缩纹（striae distensae），通常与青春期快速生长有关。摘要将其描述为“香烟烧伤”或“热损伤”与视觉事实严重不符，香烟烧伤通常是圆形的。 | 摘要在描述图片核心主体（背部的痕迹）时存在严重的误导性错误。它将典型的线性萎缩纹（生长纹/膨胀纹）错误地描述为“香烟烧伤”或“热损伤”。从视觉上看，这些痕迹是长条状、水平分布且呈紫红色的，完全不符合香烟烧伤（通常为圆形）的特征。这种对主要视觉事实的错误定性属于严重的谓词错误/语境脱离错误。 |
| 12 | 177 | entity_error | Также отображены бонусы: 118720, 9840, 5740, 2570, 0 | 错误归属：该区域标题为“Блиц-тур”而非“Бонусы”。数值错误：98960被写成9840，57440被写成5740，25700被写成2570。 | 摘要包含严重的幻觉和事实错误。它将“Блиц-тур”（闪电赛）误认为“Бонусы”（奖金），且其中的数值多处错误（如98960写成9840）；球员数量统计错误（实际32人，摘要称28人）；球员位置和姓名大量造假（如出现图中没有的“Диего Марадона”）；最严重的是摘要末尾陷入了无意义的重复循环（“Хосе Мария”重复数十次），属于严重的模型崩溃表现。 |
| 13 | 197 | entity_error | В комиссию поступил протокол от Адамин. правоборцу (вероятно, «Администрации правобережья» или аналогичного органа) в отношении Вас по факту: | 图中文字是“адм. правонаруш.”（行政违法行为），而非“Адамин. правоборцу”。 | 摘要在识别手写文字方面存在多处严重错误，导致核心事实被歪曲。主要错误包括：将“КДН”（未成年人事务委员会）误读为“КОН”；将“адм. правонаруш.”（行政违法行为）误读为“Адамин. правоборцу”；将地址“14а”误读为“№ 2”；最严重的是完全遗漏了“饮酒”这一违法事实，并错误地将“携带证件的要求”解读为“违法事实”（即所谓的缺少证件）。 |
| 14 | 209 | entity_error | - **Комментарии**: Под видео размещён комментарий на русском языке: «Победить инициаторов войны, Украине — уважение!» с лайками (40). | 错误。该评论原文为中文“打败战争发动者，向乌克兰军人致敬！”，而非俄语。 | 摘要在识别语言方面存在严重错误。它声称主标题文字和评论是俄语，但实际上图中醒目的黄色标题文字和下方评论均为中文。虽然界面 UI 是俄语，但摘要对核心内容语言的误判属于严重的实体错误（entity_error）。 |

<details><summary>展开各 badcase 完整理由</summary>

**step150-ru-case15** (准确性=2, 类型=out_of_context_error)
- 错句: Стекло лобового окна имеет трещины.
- 说明: 图中挡风玻璃上可见的是树木和电线的倒影，并无明确可见的裂纹。
- 完整理由: 摘要存在多处与图片事实明确矛盾的硬错误：1. 错误描述车轮为“无轮毂”（без дисков），而图中清晰可见铝合金轮毂；2. 错误识别地区代码，155属于俄罗斯奥姆斯科州（Omsk Oblast），而非巴什科尔托斯坦共和国；3. 错误解读UI功能，将“显示更多”（Показать ещё）这一展开正文的按钮误认为作者的请求；4. 错误描述车标为“菱形”（ромб），FAW（一汽）的车标是带翅膀的“1”字圆形标识；5. 凭空捏造了挡风玻璃有裂纹的信息（图中仅可见树木倒影）。

**step150-ru-case77** (准确性=2, 类型=entity_error)
- 错句: Он стоит по пояс в прозрачной воде...
- 说明: 错误描述了程度。图中水深仅到男孩膝盖或大腿位置，并未达到腰部。
- 完整理由: 摘要整体描述较为详细，但存在两处明显的与图片事实矛盾的错误：一是将水深描述为“齐腰”（по пояс），而图中水面仅到男孩膝盖或大腿中部；二是将手串描述在“左手”（на левой руке），而图中手串明显戴在举起做手势的右手上。根据评测标准，这类方位和程度的明确错误属于 entity_error，导致准确性评分为 2。

**step150-ru-case86** (准确性=2, 类型=entity_error)
- 错句: Название видео: «Прожил Пятую Ночь с ПИВОЗАВРОМ в ПОДЪЕЗДЕ! (5 НО...» — продолжение названия обрезано, но контекст указывает на серию видео о совместном проживании с вымышленным существом-пивозавром.
- 说明: 摘要将下方推荐视频的标题误认为当前播放视频的标题。当前视频标题为“Прожил Пять Ночей с ПИВОЗАВРОМ в...”。
- 完整理由: 摘要存在多处严重的数值与对象归属错误。首先，它将主视频的标题与下方推荐视频的标题搞混；其次，将主视频的播放量（538k）错误地描述为点赞数（图中点赞数为11k）；再次，将推荐视频的播放量（358k）说成是当前视频的播放量。此外，摘要包含大量关于“受众”、“文化背景”和“发展潜力”的凭空臆断，属于严重的幻觉信息。

**step150-ru-case106** (准确性=2, 类型=predicate_error)
- 错句: Пользователь отправляет видео с подписью «Тгк: Шутка круг», где показан человек в желтом жилете, указывающий на строительную технику на фоне свалки.
- 说明: Ошибка в привязке контента: текст «Тгк: Шутка круг» находится на первом видео (отправленном пользователем «Вы»), а человек в желтом жилете показан на втором видео (отправленном «швепс;)»).
- 完整理由: В аннотации допущены серьезные фактические ошибки: содержание двух разных видео было смешано (текст «Тгк: Шутка круг» относится к первому видео, а человек в жилете — ко второму); неверно указан эмодзи в водяном знаке (🥴 вместо 😂); неверно указано количество сообщений в форме собак (их три, а не два).

**step150-ru-case122** (准确性=2, 类型=entity_error)
- 错句: Изображение представляет собой скриншот из социальной сети, где пользователь shahrom_journalist опубликовал пост с изображением экрана смартфона, демонстрирующего чат-интерфейс с уведомлениями и сообщениями на русском языке.
- 说明: 聊天消息本身是塔吉克语，只有系统通知是俄语。
- 完整理由: 摘要存在多处严重的客观事实错误：1. 错误识别了系统通知内容，将加密通知误认为“服务不可用”；2. 错误识别了对话双方，将回复方误认为同一人；3. 错误转录了聊天文字并提供了错误的翻译；4. 遗漏了图中最重要的信息（索要1500美元的诈骗内容）；5. 错误识别了时间（1:43误为14:43）和表情符号（侦探误为小丑）。

**step150-ru-case126** (准确性=2, 类型=entity_error)
- 错句: Изображение представляет собой страницы из книги А. Корнева «Магия. Сакральные обряды и ритуалы. Практическое руководство», посвящённые двум магическим ритуалам: «Полный карман» и «Заговоренная купюра».
- 说明: В названии книги на фото написано «Сильнейшие обряды», а не «Сакральные».
- 完整理由: Сводка содержит множество фактических ошибок и галлюцинаций. Во-первых, неверно указано название книги («Сакральные» вместо «Сильнейшие»). Во-вторых, в описании первого ритуала допущены грубые ошибки: текст предписывает нести купюру в банк, а не класть в кошелек, и распределять монеты разными способами, а не оставлять все в кошельке. В-третьих, в истории про торговку роли перепутаны: это она просила помощи у автора, а не помогала ему. Кроме того, большая часть разделов «Общие рекомендации» и «Дополнительные детали» (про семь планет, чистоту купюры, запрет рассказывать о ритуале и время проведения вечером/ночью) полностью выдумана и отсутствует в тексте на изображении.

**step150-ru-case128** (准确性=2, 类型=entity_error)
- 错句: Монтажники любого профиля: от 18 000 до 20 000 рублей в месяц.
- 说明: 图中数值为“180000”（18万），摘要少写了一个零且编造了范围。
- 完整理由: 摘要在关键数值和材料描述上存在严重错误。首先，将安装工（монтажники）的工资“180000”（18万）错误地写成了“18 000 до 20 000”（1.8万到2万），数值相差一个数量级；其次，摘要声称焊接结构包含“不锈钢”（нержавеющей），而图中明确说明是“普通黑色金属”（обычный черновой металл），这属于凭空捏造的错误信息。

**step150-ru-case131** (准确性=2, 类型=entity_error)
- 错句: Решение проблем с недвижимостью (khandazmuli, vadagadacilebuli an gaproblemebuli seskhi).
- 说明: 原文中的 'seskhi' 意为 '贷款'（loan），而非 '房地产'（недвижимость）。
- 完整理由: 摘要在翻译和理解格鲁吉亚语原文时存在多处严重错误。首先，将“seskhi”（贷款）错误地翻译为“недвижимость”（房地产）；其次，将“nebismier”（任何）臆断为“бесплатных”（免费的）；此外，摘要误解了文中提到的“kompaniebi”（公司），原文是指催收公司在骚扰用户，而非为公司提供办证服务。这些核心业务信息的错误属于严重的实体和谓词错误。

**step150-ru-case145** (准确性=2, 类型=predicate_error)
- 错句: Медиа-контент: В начале переписки — видео длительностью 01:01, за которым следует исходящий аудиовызов длительностью 30:48 (отправлен в 01:07).
- 说明: 图中“01:01”是消息发送的时间戳，而非视频时长。
- 完整理由: 摘要在关键事实描述上存在多处明确错误：1. 将消息发送时间“01:01”误认为视频时长；2. 错误归属了消息发送方，将用户发送的“Иди пей”说成是 Julia 发送的；3. 混淆了通话时长与时间戳，将来电时长（10:53）与时间（01:20）写反了。

**step150-ru-case158** (准确性=2, 类型=predicate_error)
- 错句: На изображении представлен скриншот интерфейса чата в приложении, где пользователь Ignat отправляет эмоциональное сообщение о предательстве и душевной боли, а ниже — ответ от ИИ с юмористическим и ироничным подтекстом.
- 说明: 角色识别错误。Ignat 是 AI 角色，其发送的是感性消息；下方的荒诞消息是用户发送的，而非 AI。
- 完整理由: 摘要在识别对话角色方面存在严重的根本性错误。在 Character.ai 界面中，顶部的名字“Ignat”是 AI 角色的名字，其下方的感性文字是 AI 的回复；而底部带有通用头像的消息（“*взял пистолет и убился*...”）是用户的输入或发送的消息。摘要完全颠倒了这两者，将 Ignat 称为“用户”，将底部的荒诞消息称为“AI 的回复”。这种角色错位导致了后续关于情感基调、心理分析和互动逻辑的所有推论都是错误的。

**step150-ru-case160** (准确性=2, 类型=predicate_error)
- 错句: На коже спины видны множественные линейные и волнистые следы, напоминающие ожоги от сигареты или другие термические повреждения.
- 说明: 这是严重的错误。图中的痕迹是典型的水平线性萎缩纹（striae distensae），通常与青春期快速生长有关。摘要将其描述为“香烟烧伤”或“热损伤”与视觉事实严重不符，香烟烧伤通常是圆形的。
- 完整理由: 摘要在描述图片核心主体（背部的痕迹）时存在严重的误导性错误。它将典型的线性萎缩纹（生长纹/膨胀纹）错误地描述为“香烟烧伤”或“热损伤”。从视觉上看，这些痕迹是长条状、水平分布且呈紫红色的，完全不符合香烟烧伤（通常为圆形）的特征。这种对主要视觉事实的错误定性属于严重的谓词错误/语境脱离错误。

**step150-ru-case177** (准确性=1, 类型=entity_error)
- 错句: Также отображены бонусы: 118720, 9840, 5740, 2570, 0
- 说明: 错误归属：该区域标题为“Блиц-тур”而非“Бонусы”。数值错误：98960被写成9840，57440被写成5740，25700被写成2570。
- 完整理由: 摘要包含严重的幻觉和事实错误。它将“Блиц-тур”（闪电赛）误认为“Бонусы”（奖金），且其中的数值多处错误（如98960写成9840）；球员数量统计错误（实际32人，摘要称28人）；球员位置和姓名大量造假（如出现图中没有的“Диего Марадона”）；最严重的是摘要末尾陷入了无意义的重复循环（“Хосе Мария”重复数十次），属于严重的模型崩溃表现。

**step150-ru-case197** (准确性=2, 类型=entity_error)
- 错句: В комиссию поступил протокол от Адамин. правоборцу (вероятно, «Администрации правобережья» или аналогичного органа) в отношении Вас по факту:
- 说明: 图中文字是“адм. правонаруш.”（行政违法行为），而非“Адамин. правоборцу”。
- 完整理由: 摘要在识别手写文字方面存在多处严重错误，导致核心事实被歪曲。主要错误包括：将“КДН”（未成年人事务委员会）误读为“КОН”；将“адм. правонаруш.”（行政违法行为）误读为“Адамин. правоборцу”；将地址“14а”误读为“№ 2”；最严重的是完全遗漏了“饮酒”这一违法事实，并错误地将“携带证件的要求”解读为“违法事实”（即所谓的缺少证件）。

**step150-ru-case209** (准确性=2, 类型=entity_error)
- 错句: - **Комментарии**: Под видео размещён комментарий на русском языке: «Победить инициаторов войны, Украине — уважение!» с лайками (40).
- 说明: 错误。该评论原文为中文“打败战争发动者，向乌克兰军人致敬！”，而非俄语。
- 完整理由: 摘要在识别语言方面存在严重错误。它声称主标题文字和评论是俄语，但实际上图中醒目的黄色标题文字和下方评论均为中文。虽然界面 UI 是俄语，但摘要对核心内容语言的误判属于严重的实体错误（entity_error）。

</details>

#### zh (中文) — 23 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 17 | entity_error | 聊天界面顶部显示时间为周六下午2:24至5:09，背景壁纸为电影《荒野猎人》画面。 | 背景壁纸并非《荒野猎人》，而是《最后生还者》（The Last of Us）中乔尔和艾莉的形象。 | 摘要存在多处明显的实体和谓词错误。首先，背景壁纸被错误识别为电影《荒野猎人》，实际上是美剧《最后生还者》（The Last of Us）的剧照。其次，对聊天内容的翻译存在严重偏差：“Naa dw ingun roda”中的“roda”在此语境下是人名，意为“Roda说有”，摘要将其误译为“不是那个轮子”；“Puti dw to”意为“说是白色的”，摘要将其误译为“不是那个白色的”，将肯定句误解为否定句。 |
| 2 | 36 | entity_error | 4. **界面信息**：截图包含典型的短视频平台UI元素，左侧显示点赞数（105.3万）、评论数（1645）及转发数（910）；底部文字显示账号名称为“Zainab Sabah”，并包含阿拉伯语互动话题标签。 | 图中点赞数显示为“105.3 ألف”（阿拉伯语“ألف”意为千），即10.53万，摘要写成“105.3万”，存在数量级错误。其余数值和文字信息正确。 | 摘要在描述社交媒体互动数据时存在严重的数量级错误。图中显示的点赞数为“105.3 ألف”（即 105.3 千，等同于 10.53 万），而摘要将其错误地写为“105.3万”，数值扩大了10倍，属于明确的实体错误（entity_error）。其余关于人物、服饰及背景的描述与事实相符。 |
| 3 | 54 | entity_error | 意为“我们要去当雨天的父母了哦！！” | 严重的翻译错误。“magpares”在菲律宾语境中指去吃 Pares（一种牛肉炖菜），而非“当父母”（父母是 magulang）。这属于严重的语义幻觉，误导了对动态核心含义的理解。 | 摘要在描述核心动态内容时存在严重的事实性错误。虽然界面和背景描述准确，但对配文的翻译属于严重的语义幻觉：将“去吃 Pares（一种菲律宾炖牛肉）”误解为“当父母”（entity_error），这完全扭曲了发布者的意图。图片背景中的 Coca-Cola 冰箱和餐饮环境也佐证了其内容与食物有关，而非家庭身份。此外，配文转录中存在单字符拼写错误（magparens vs magpares）。 |
| 4 | 55 | circumstantial_error | 这是一张手机聊天软件（界面特征符合Telegram）的截图，展示了用户与名为“Апаи Саида”（阿帕伊·赛达）的联系人之间的对话记录。 | 界面特征（如绿色顶栏、附件图标、状态回复机制）明确显示这是 WhatsApp，而非 Telegram。 | 摘要存在多处与图片事实明确矛盾的硬错误：1. 界面识别错误，图中明显的绿色调、图标样式及“Статус”（状态）功能均指向 WhatsApp，而非 Telegram；2. 消息类型与归属错误，摘要将用户发布的“状态更新（Status）”误认为“发送的语音消息”，且错误声称对方回复了语音消息（图中对方仅回复了表情和文字）；3. 语言识别错误，对话文字为塔吉克语（Tajik），而非俄语或俄语混合语。 |
| 5 | 64 | entity_error | **互动数据**：视频右侧显示点赞数为90.7万，评论数为8.7千，转发数为2.5万，收藏数为7.9千。 | 图片中收藏数（书签图标）显示为79K，即7.9万，摘要错误地写成了7.9千，数值缩小了10倍。 | 摘要整体上忠实描述了图片内容，但在互动数据部分存在一处明显的数值错误：图片中收藏数（书签图标）显示为“79K”（即7.9万），而摘要将其写为“7.9千”，数值相差一个数量级，属于实质性的实体错误。 |
| 6 | 74 | entity_error | 日期标注为 2020年4月16日。 | 图片中支票上的日期明确标注为“April 10, 2024”，摘要中的年份（2020）和日期（16日）均错误。 | 摘要在关键数据描述上存在严重的事实性错误。支票上的日期清晰显示为“April 10, 2024”，而摘要将其写为“2020年4月16日”，年份和日期均与图片事实不符，属于 entity_error。其余部分描述准确。 |
| 7 | 80 | entity_error | 游戏角色：视频画面主要展示了《Mobile Legends》中的英雄角色“Eudora”（艾朵拉）的多个皮肤，包括“Night Shade”、“Christmas”、“Plunderous Pirate”等。 | “Night Shade”是图中Ling的皮肤，“Christmas”是Gord的皮肤，图中Eudora仅展示了一个皮肤“Lightning Magician”。 | 摘要存在多处严重的实体错误。首先，它错误地将图中其他英雄的皮肤归属于Eudora（“Night Shade”是Ling的皮肤，“Christmas”是Gord的皮肤，图中Eudora仅展示了“Lightning Magician”皮肤）；其次，它将视频上方的菲律宾语（Tagalog）文字误认为泰语。这些错误涉及图片核心内容的识别。 |
| 8 | 81 | circumstantial_error | 视频背景：背景视频为车内视角，可见车窗外有蓝色装饰物。 | 蓝色装饰物位于车内仪表盘上，而非车窗外。 | 摘要存在明显的翻译错误和事实误导。用户“atayev09”评论的“aydymyn ady name”在土库曼语中意为“这首歌叫什么名字”，而非摘要中所称的“我的名字是什么”；因此，该用户是在询问背景音乐名称，而非询问作者身份。此外，蓝色装饰物位于车内仪表盘上，而非车窗外。 |
| 9 | 85 | entity_error | 这是一张即时通讯软件（Telegram）的聊天界面截图 | 图中 UI 元素（如回形针附件图标、相机图标、绿色发送/语音按钮）是典型的 WhatsApp 界面，而非 Telegram。 | 摘要存在多处实质性事实错误：1. 界面识别错误，图中 UI（如附件图标、发送按钮样式）明显属于 WhatsApp 而非 Telegram；2. 语言识别及翻译错误，消息原文为塔吉克语而非乌兹别克语，且内容意为“我打了几次电话都没打通”，而非摘要所述的“拍了视频”；3. 时间归属错误，语音通话记录在“今天（Сегодня）”栏目下，而非摘要所述的“当天（7月20日）”。 |
| 10 | 96 | entity_error | 消息内容包含阿姆哈拉语文字，其中一条带有WhatsApp图标，另一条带有Discord图标。 | 图标识别错误。第一条通知的图标是 Telegram，第二条是 Canva，而非摘要所述的 WhatsApp 和 Discord。 | 摘要在识别通知来源的应用图标时存在严重的事实性错误。图中第一条通知的图标明显是 Telegram（蓝色圆圈内含白色纸飞机），第二条通知的图标是 Canva（黑色圆圈内含文字），而摘要将其错误地描述为 WhatsApp 和 Discord。这属于典型的实体错误（entity_error）。 |
| 11 | 101 | circumstantial_error | 这是一张名为“Kelly Anibor”的微信聊天界面截图，记录了双方关于通话未接及后续沟通的对话。 | 图片显示的 UI 界面（绿色气泡、图标样式、布局）明显是 WhatsApp，而非微信。 | 摘要存在两处明显的硬伤：首先，将界面类型错误识别为“微信”（实际上是 WhatsApp）；其次，在沟通内容的描述中，将“Fine and you”这一消息的发送方搞错（图中绿色气泡显示是用户发送的，而非对方 Kelly 发送的）。 |
| 12 | 108 | entity_error | 画面中可见大型黄色挖掘机正在向一辆长条卡车装载碎石 | 挖掘机实际上位于水中的驳船上，动作是向岸边卸石，而非向卡车装载。 | 摘要存在两处明确的事实错误：1. 画面中挖掘机位于水面的驳船上，正在向岸边卸载碎石，而非摘要所述的“向长条卡车装载”，这属于实体与谓词错误；2. 点赞数“24.7万”与图中显示的“24,7 mil”（即2.47万）相比，数值差了一个数量级，属于严重的实体错误。根据标准，数量级差异及主体动作/对象错误应评为2分。 |
| 13 | 128 | entity_error | - **人物形象**：画面主体为身着西装的男性（马克·泽尔曼），表情严肃，双手交叠置于桌前，背景为会议室环境。 | 图中人物是迈克尔·科恩（Michael Cohen），而非“马克·泽尔曼”，属于实体识别错误。 | 摘要主体内容基本属实，但存在一处严重的实体错误（entity_error）：将画面中的人物（迈克尔·科恩 Michael Cohen）错误识别为“马克·泽尔曼”。这种凭空捏造且与事实不符的姓名属于实质性错误。 |
| 14 | 145 | circumstantial_error | 这是一张微信聊天截图，记录了用户与名为“Ian Joshua Cariño”的联系人关于租房及付款事宜的对话。 | 界面是 Facebook Messenger 而非微信；对话主题是借贷利息而非租房。 | 摘要存在多处严重的实质性错误。首先，界面被错误识别为“微信”（实际为 Facebook Messenger）。其次，摘要完全误解了对话的主题：对话中使用塔加洛语讨论的是民间借贷及利息（tubo 指利息，papahiram 指出借），而非“租房”。摘要将“利息”误认为“租金”或“押金”，将“20%的利率”误认为“折扣”，并凭空捏造了“房屋出租”的情节。虽然识别出了人物姓名和摩托车附件，但核心事实完全错误。 |
| 15 | 147 | entity_error | 付款账户：1897365413 | 错误。1897365413 是“Рақами амалиёт”（交易单号），而非付款账户。 | 摘要在关键数据归属上存在严重错误。它将“交易单号”（1897365413）误认为“付款账户”，并且完全颠倒了付款账户（9762***5094）与收款账户（992186303030）的身份。这种实质性的实体错误和逻辑颠倒严重影响了信息的准确性。 |
| 16 | 166 | entity_error | 这是一张短视频平台的截图，内容涉及台湾政治人物戴瑗姍的争议性言论。 | 图中文字明确显示人物姓名为“戴瑋姍”，摘要误写为“戴瑗姍”。 | 摘要存在严重的实体错误，将图中主体人物的姓名“戴瑋姍”全程错误识别为“戴瑗姍”（“瑋”与“瑗”字形不同）。此外，摘要将视频上方的批评性标题文案（“台湾海域被瓜分竟然叫好”）直接归为人物的言论，存在一定的误导性。 |
| 17 | 168 | entity_error | 发布者：账号名为“አዲስ ማህበረሰብ”（Adeis Mahiberesib），发布时间为22小时前。 | 账号名错误。图中文字为“አዲስ መረጃ”（Addis Mereja，意为新信息/新闻），而非摘要所述名称。 | 摘要在核心事实识别上存在严重错误。首先，发布者账号名识别错误（图中为“አዲስ መረጃ”，摘要误写为“አዲስ ማህበረሰብ”）；其次，摘要完全误读了帖子的文字内容，将“政府官员任命新闻”（#ዜና_ሹመት，意为任命新闻）臆断为“婚礼”，并凭空捏造了图中不存在的埃塞俄比亚语标签和词汇含义。虽然画面视觉描述基本正确，但对文字信息的错误解读导致了主体事实的颠倒。 |
| 18 | 173 | entity_error | 库里亚呼吁执政党（UDA）将原本用于支持反对党（DCP）的“资金”（mitungi）收回，并指出当前已选出DCP议员。 | 图中文字“mitungi ya gas”意为“煤气罐”，而非“资金”；且原文是让民众退回 UDA 的煤气罐去领 DCP 的，而非呼吁执政党收回资金。 | 摘要在核心内容的解读上存在严重的事实性错误。首先，将图中明确提到的“mitungi ya gas”（煤气罐）错误地表述为“资金”，属于实体错误；其次，摘要臆断库里亚主张“双方应坐下来谈判”，而图中文字实际上是对选民的讽刺性挑战，质疑新当选的反对党议员是否能兑现发展承诺，完全未提及谈判，属于语境脱离错误。 |
| 19 | 176 | entity_error | 埃塞俄比亚广播公司发布关于2025年4月25日（埃塞俄比亚历法）的官方声明，并配发了相关会议现场图片。 | 图中文字显示的是“25.4 ቢሊዮን ብር”（25.4亿比尔）的预算，而非“2025年4月25日”这一日期。 | 摘要存在严重的实体错误。它将图中文字提到的“25.4亿比尔（25.4 ቢሊዮን ብር）”预算金额错误地解读为“2025年4月25日”这一日期，导致对帖文核心主题（预算审批）的描述完全偏离事实。虽然对画面视觉元素的描述（会议场景、人物着装、投票动作）是正确的，但核心信息的错误属于实质性误导。 |
| 20 | 185 | entity_error | 标题：阿拉伯语显示为“热力学第一定律 第4课”，由“哈希姆·阿尔-盖拉比”（Hashim Al-Gharabi）教授主讲。 | 图中视频标题原文为“دالة الحالة و دالة المسار \|\| الفصل الأول الثرموداينمك”（状态函数与路径函数 \|\| 第一章 热力学），摘要中的标题内容与原文不符。 | 摘要在关键事实描述上存在多处严重错误。首先，数值单位理解错误，将“45 ألف”（4.5万）误记为“45万”，将“8.9 ألف”（8900）误记为“8.9万”，数量级偏差达10倍；其次，视频标题翻译/识别有误，将“电子课程”（الدورات الإلكترونية）误认为“电子电路”，且第一个视频的标题内容与原图文字（状态函数与路径函数）不符。这些属于明确的实体错误和幻觉。 |
| 21 | 186 | entity_error | 截图展示了一个名为“Nile Quiz”的埃塞俄比亚语问答视频，标题涉及“婚礼”（ገናታሪ）和“新娘”（ገናታሪ）等词汇，属于语言学习或趣味问答类内容。 | 图中并未出现“ገናታሪ”这个词。图中表示婚礼的词是“ሰርግ”，表示谜题的词是“እንቆቅልሽ”。摘要凭空捏造了文字内容。 | 摘要在描述图内文字时存在多处严重的实体错误和幻觉。它反复提到图中不存在的词汇“ገናታሪ”，并将其错误地解释为“婚礼”或“新娘”（图中实际的婚礼词汇是“ሰርግ”）。此外，它将视频封面上的标题“ሰርግ ተበላሸ!”（婚礼被毁了！）错误地描述为“ገናታሪ ተሰጥቶ!”（婚礼开始了！），这与图片事实完全相反。 |
| 22 | 213 | predicate_error | 用户向“Mr Ako'o”索要其妻子的牧师联系方式，随后“Mr Ako'o”回复“Abbe Bindzi”并发送了消息。 | 动作主体颠倒且事实错误：是Mr Ako'o（左侧）发消息给右侧用户索要神父（Prêtre）的号码，以便和他妻子一起去见神父；右侧用户随后回复了Abbé Bindzi的联系方式。 | 摘要在核心事件的动作主体和对话逻辑上存在严重错误。首先，索要号码的是Mr Ako'o（左侧灰色气泡），而非摘要所说的“用户向Mr Ako'o索要”；其次，索要的是“神父（Prêtre）”的号码，而非“妻子的牧师”；最后，要求宣布结婚和给一箱啤酒的是右侧用户，而非Mr Ako'o。这些错误导致对话逻辑完全颠倒。 |
| 23 | 218 | entity_error | 利比亚多地正经历热浪，气温在48至50摄氏度之间波动，部分地区甚至达到43度。 | 图中文字说明气温普遍在43-48度，特定地区（阿齐齐亚和贾法拉）可达49-50度。摘要将43度描述为“甚至达到”，逻辑错误，且数值范围描述不精确。 | 摘要存在多处与图片事实不符的硬伤：1. 虚构了“埃及（尼罗河地区）”这一地点，图中文字仅提及利比亚的 العزيزية（阿齐齐亚）和 جفارة（贾法拉）；2. 错误归属了天气转折的时间，图中明确说是“周二（الثلاثاء）”下午开始降温，摘要写成了“周三”；3. 对气温数值的逻辑描述有误，图中称气温在43-48度之间，局部可达49-50度，摘要却称“甚至达到43度”，将低值误作极端高值；4. 凭空捏造了“早晚温差大，夜间依然炎热”的信息，图中并无此类描述。 |

<details><summary>展开各 badcase 完整理由</summary>

**step150-zh-case17** (准确性=2, 类型=entity_error)
- 错句: 聊天界面顶部显示时间为周六下午2:24至5:09，背景壁纸为电影《荒野猎人》画面。
- 说明: 背景壁纸并非《荒野猎人》，而是《最后生还者》（The Last of Us）中乔尔和艾莉的形象。
- 完整理由: 摘要存在多处明显的实体和谓词错误。首先，背景壁纸被错误识别为电影《荒野猎人》，实际上是美剧《最后生还者》（The Last of Us）的剧照。其次，对聊天内容的翻译存在严重偏差：“Naa dw ingun roda”中的“roda”在此语境下是人名，意为“Roda说有”，摘要将其误译为“不是那个轮子”；“Puti dw to”意为“说是白色的”，摘要将其误译为“不是那个白色的”，将肯定句误解为否定句。

**step150-zh-case36** (准确性=2, 类型=entity_error)
- 错句: 4. **界面信息**：截图包含典型的短视频平台UI元素，左侧显示点赞数（105.3万）、评论数（1645）及转发数（910）；底部文字显示账号名称为“Zainab Sabah”，并包含阿拉伯语互动话题标签。
- 说明: 图中点赞数显示为“105.3 ألف”（阿拉伯语“ألف”意为千），即10.53万，摘要写成“105.3万”，存在数量级错误。其余数值和文字信息正确。
- 完整理由: 摘要在描述社交媒体互动数据时存在严重的数量级错误。图中显示的点赞数为“105.3 ألف”（即 105.3 千，等同于 10.53 万），而摘要将其错误地写为“105.3万”，数值扩大了10倍，属于明确的实体错误（entity_error）。其余关于人物、服饰及背景的描述与事实相符。

**step150-zh-case54** (准确性=2, 类型=entity_error)
- 错句: 意为“我们要去当雨天的父母了哦！！”
- 说明: 严重的翻译错误。“magpares”在菲律宾语境中指去吃 Pares（一种牛肉炖菜），而非“当父母”（父母是 magulang）。这属于严重的语义幻觉，误导了对动态核心含义的理解。
- 完整理由: 摘要在描述核心动态内容时存在严重的事实性错误。虽然界面和背景描述准确，但对配文的翻译属于严重的语义幻觉：将“去吃 Pares（一种菲律宾炖牛肉）”误解为“当父母”（entity_error），这完全扭曲了发布者的意图。图片背景中的 Coca-Cola 冰箱和餐饮环境也佐证了其内容与食物有关，而非家庭身份。此外，配文转录中存在单字符拼写错误（magparens vs magpares）。

**step150-zh-case55** (准确性=2, 类型=circumstantial_error)
- 错句: 这是一张手机聊天软件（界面特征符合Telegram）的截图，展示了用户与名为“Апаи Саида”（阿帕伊·赛达）的联系人之间的对话记录。
- 说明: 界面特征（如绿色顶栏、附件图标、状态回复机制）明确显示这是 WhatsApp，而非 Telegram。
- 完整理由: 摘要存在多处与图片事实明确矛盾的硬错误：1. 界面识别错误，图中明显的绿色调、图标样式及“Статус”（状态）功能均指向 WhatsApp，而非 Telegram；2. 消息类型与归属错误，摘要将用户发布的“状态更新（Status）”误认为“发送的语音消息”，且错误声称对方回复了语音消息（图中对方仅回复了表情和文字）；3. 语言识别错误，对话文字为塔吉克语（Tajik），而非俄语或俄语混合语。

**step150-zh-case64** (准确性=2, 类型=entity_error)
- 错句: **互动数据**：视频右侧显示点赞数为90.7万，评论数为8.7千，转发数为2.5万，收藏数为7.9千。
- 说明: 图片中收藏数（书签图标）显示为79K，即7.9万，摘要错误地写成了7.9千，数值缩小了10倍。
- 完整理由: 摘要整体上忠实描述了图片内容，但在互动数据部分存在一处明显的数值错误：图片中收藏数（书签图标）显示为“79K”（即7.9万），而摘要将其写为“7.9千”，数值相差一个数量级，属于实质性的实体错误。

**step150-zh-case74** (准确性=2, 类型=entity_error)
- 错句: 日期标注为 2020年4月16日。
- 说明: 图片中支票上的日期明确标注为“April 10, 2024”，摘要中的年份（2020）和日期（16日）均错误。
- 完整理由: 摘要在关键数据描述上存在严重的事实性错误。支票上的日期清晰显示为“April 10, 2024”，而摘要将其写为“2020年4月16日”，年份和日期均与图片事实不符，属于 entity_error。其余部分描述准确。

**step150-zh-case80** (准确性=2, 类型=entity_error)
- 错句: 游戏角色：视频画面主要展示了《Mobile Legends》中的英雄角色“Eudora”（艾朵拉）的多个皮肤，包括“Night Shade”、“Christmas”、“Plunderous Pirate”等。
- 说明: “Night Shade”是图中Ling的皮肤，“Christmas”是Gord的皮肤，图中Eudora仅展示了一个皮肤“Lightning Magician”。
- 完整理由: 摘要存在多处严重的实体错误。首先，它错误地将图中其他英雄的皮肤归属于Eudora（“Night Shade”是Ling的皮肤，“Christmas”是Gord的皮肤，图中Eudora仅展示了“Lightning Magician”皮肤）；其次，它将视频上方的菲律宾语（Tagalog）文字误认为泰语。这些错误涉及图片核心内容的识别。

**step150-zh-case81** (准确性=2, 类型=circumstantial_error)
- 错句: 视频背景：背景视频为车内视角，可见车窗外有蓝色装饰物。
- 说明: 蓝色装饰物位于车内仪表盘上，而非车窗外。
- 完整理由: 摘要存在明显的翻译错误和事实误导。用户“atayev09”评论的“aydymyn ady name”在土库曼语中意为“这首歌叫什么名字”，而非摘要中所称的“我的名字是什么”；因此，该用户是在询问背景音乐名称，而非询问作者身份。此外，蓝色装饰物位于车内仪表盘上，而非车窗外。

**step150-zh-case85** (准确性=2, 类型=entity_error)
- 错句: 这是一张即时通讯软件（Telegram）的聊天界面截图
- 说明: 图中 UI 元素（如回形针附件图标、相机图标、绿色发送/语音按钮）是典型的 WhatsApp 界面，而非 Telegram。
- 完整理由: 摘要存在多处实质性事实错误：1. 界面识别错误，图中 UI（如附件图标、发送按钮样式）明显属于 WhatsApp 而非 Telegram；2. 语言识别及翻译错误，消息原文为塔吉克语而非乌兹别克语，且内容意为“我打了几次电话都没打通”，而非摘要所述的“拍了视频”；3. 时间归属错误，语音通话记录在“今天（Сегодня）”栏目下，而非摘要所述的“当天（7月20日）”。

**step150-zh-case96** (准确性=2, 类型=entity_error)
- 错句: 消息内容包含阿姆哈拉语文字，其中一条带有WhatsApp图标，另一条带有Discord图标。
- 说明: 图标识别错误。第一条通知的图标是 Telegram，第二条是 Canva，而非摘要所述的 WhatsApp 和 Discord。
- 完整理由: 摘要在识别通知来源的应用图标时存在严重的事实性错误。图中第一条通知的图标明显是 Telegram（蓝色圆圈内含白色纸飞机），第二条通知的图标是 Canva（黑色圆圈内含文字），而摘要将其错误地描述为 WhatsApp 和 Discord。这属于典型的实体错误（entity_error）。

**step150-zh-case101** (准确性=2, 类型=circumstantial_error)
- 错句: 这是一张名为“Kelly Anibor”的微信聊天界面截图，记录了双方关于通话未接及后续沟通的对话。
- 说明: 图片显示的 UI 界面（绿色气泡、图标样式、布局）明显是 WhatsApp，而非微信。
- 完整理由: 摘要存在两处明显的硬伤：首先，将界面类型错误识别为“微信”（实际上是 WhatsApp）；其次，在沟通内容的描述中，将“Fine and you”这一消息的发送方搞错（图中绿色气泡显示是用户发送的，而非对方 Kelly 发送的）。

**step150-zh-case108** (准确性=2, 类型=entity_error)
- 错句: 画面中可见大型黄色挖掘机正在向一辆长条卡车装载碎石
- 说明: 挖掘机实际上位于水中的驳船上，动作是向岸边卸石，而非向卡车装载。
- 完整理由: 摘要存在两处明确的事实错误：1. 画面中挖掘机位于水面的驳船上，正在向岸边卸载碎石，而非摘要所述的“向长条卡车装载”，这属于实体与谓词错误；2. 点赞数“24.7万”与图中显示的“24,7 mil”（即2.47万）相比，数值差了一个数量级，属于严重的实体错误。根据标准，数量级差异及主体动作/对象错误应评为2分。

**step150-zh-case128** (准确性=2, 类型=entity_error)
- 错句: - **人物形象**：画面主体为身着西装的男性（马克·泽尔曼），表情严肃，双手交叠置于桌前，背景为会议室环境。
- 说明: 图中人物是迈克尔·科恩（Michael Cohen），而非“马克·泽尔曼”，属于实体识别错误。
- 完整理由: 摘要主体内容基本属实，但存在一处严重的实体错误（entity_error）：将画面中的人物（迈克尔·科恩 Michael Cohen）错误识别为“马克·泽尔曼”。这种凭空捏造且与事实不符的姓名属于实质性错误。

**step150-zh-case145** (准确性=2, 类型=circumstantial_error)
- 错句: 这是一张微信聊天截图，记录了用户与名为“Ian Joshua Cariño”的联系人关于租房及付款事宜的对话。
- 说明: 界面是 Facebook Messenger 而非微信；对话主题是借贷利息而非租房。
- 完整理由: 摘要存在多处严重的实质性错误。首先，界面被错误识别为“微信”（实际为 Facebook Messenger）。其次，摘要完全误解了对话的主题：对话中使用塔加洛语讨论的是民间借贷及利息（tubo 指利息，papahiram 指出借），而非“租房”。摘要将“利息”误认为“租金”或“押金”，将“20%的利率”误认为“折扣”，并凭空捏造了“房屋出租”的情节。虽然识别出了人物姓名和摩托车附件，但核心事实完全错误。

**step150-zh-case147** (准确性=2, 类型=entity_error)
- 错句: 付款账户：1897365413
- 说明: 错误。1897365413 是“Рақами амалиёт”（交易单号），而非付款账户。
- 完整理由: 摘要在关键数据归属上存在严重错误。它将“交易单号”（1897365413）误认为“付款账户”，并且完全颠倒了付款账户（9762***5094）与收款账户（992186303030）的身份。这种实质性的实体错误和逻辑颠倒严重影响了信息的准确性。

**step150-zh-case166** (准确性=2, 类型=entity_error)
- 错句: 这是一张短视频平台的截图，内容涉及台湾政治人物戴瑗姍的争议性言论。
- 说明: 图中文字明确显示人物姓名为“戴瑋姍”，摘要误写为“戴瑗姍”。
- 完整理由: 摘要存在严重的实体错误，将图中主体人物的姓名“戴瑋姍”全程错误识别为“戴瑗姍”（“瑋”与“瑗”字形不同）。此外，摘要将视频上方的批评性标题文案（“台湾海域被瓜分竟然叫好”）直接归为人物的言论，存在一定的误导性。

**step150-zh-case168** (准确性=2, 类型=entity_error)
- 错句: 发布者：账号名为“አዲስ ማህበረሰብ”（Adeis Mahiberesib），发布时间为22小时前。
- 说明: 账号名错误。图中文字为“አዲስ መረጃ”（Addis Mereja，意为新信息/新闻），而非摘要所述名称。
- 完整理由: 摘要在核心事实识别上存在严重错误。首先，发布者账号名识别错误（图中为“አዲስ መረጃ”，摘要误写为“አዲስ ማህበረሰብ”）；其次，摘要完全误读了帖子的文字内容，将“政府官员任命新闻”（#ዜና_ሹመት，意为任命新闻）臆断为“婚礼”，并凭空捏造了图中不存在的埃塞俄比亚语标签和词汇含义。虽然画面视觉描述基本正确，但对文字信息的错误解读导致了主体事实的颠倒。

**step150-zh-case173** (准确性=2, 类型=entity_error)
- 错句: 库里亚呼吁执政党（UDA）将原本用于支持反对党（DCP）的“资金”（mitungi）收回，并指出当前已选出DCP议员。
- 说明: 图中文字“mitungi ya gas”意为“煤气罐”，而非“资金”；且原文是让民众退回 UDA 的煤气罐去领 DCP 的，而非呼吁执政党收回资金。
- 完整理由: 摘要在核心内容的解读上存在严重的事实性错误。首先，将图中明确提到的“mitungi ya gas”（煤气罐）错误地表述为“资金”，属于实体错误；其次，摘要臆断库里亚主张“双方应坐下来谈判”，而图中文字实际上是对选民的讽刺性挑战，质疑新当选的反对党议员是否能兑现发展承诺，完全未提及谈判，属于语境脱离错误。

**step150-zh-case176** (准确性=2, 类型=entity_error)
- 错句: 埃塞俄比亚广播公司发布关于2025年4月25日（埃塞俄比亚历法）的官方声明，并配发了相关会议现场图片。
- 说明: 图中文字显示的是“25.4 ቢሊዮን ብር”（25.4亿比尔）的预算，而非“2025年4月25日”这一日期。
- 完整理由: 摘要存在严重的实体错误。它将图中文字提到的“25.4亿比尔（25.4 ቢሊዮን ብር）”预算金额错误地解读为“2025年4月25日”这一日期，导致对帖文核心主题（预算审批）的描述完全偏离事实。虽然对画面视觉元素的描述（会议场景、人物着装、投票动作）是正确的，但核心信息的错误属于实质性误导。

**step150-zh-case185** (准确性=2, 类型=entity_error)
- 错句: 标题：阿拉伯语显示为“热力学第一定律 第4课”，由“哈希姆·阿尔-盖拉比”（Hashim Al-Gharabi）教授主讲。
- 说明: 图中视频标题原文为“دالة الحالة و دالة المسار || الفصل الأول الثرموداينمك”（状态函数与路径函数 || 第一章 热力学），摘要中的标题内容与原文不符。
- 完整理由: 摘要在关键事实描述上存在多处严重错误。首先，数值单位理解错误，将“45 ألف”（4.5万）误记为“45万”，将“8.9 ألف”（8900）误记为“8.9万”，数量级偏差达10倍；其次，视频标题翻译/识别有误，将“电子课程”（الدورات الإلكترونية）误认为“电子电路”，且第一个视频的标题内容与原图文字（状态函数与路径函数）不符。这些属于明确的实体错误和幻觉。

**step150-zh-case186** (准确性=2, 类型=entity_error)
- 错句: 截图展示了一个名为“Nile Quiz”的埃塞俄比亚语问答视频，标题涉及“婚礼”（ገናታሪ）和“新娘”（ገናታሪ）等词汇，属于语言学习或趣味问答类内容。
- 说明: 图中并未出现“ገናታሪ”这个词。图中表示婚礼的词是“ሰርግ”，表示谜题的词是“እንቆቅልሽ”。摘要凭空捏造了文字内容。
- 完整理由: 摘要在描述图内文字时存在多处严重的实体错误和幻觉。它反复提到图中不存在的词汇“ገናታሪ”，并将其错误地解释为“婚礼”或“新娘”（图中实际的婚礼词汇是“ሰርግ”）。此外，它将视频封面上的标题“ሰርግ ተበላሸ!”（婚礼被毁了！）错误地描述为“ገናታሪ ተሰጥቶ!”（婚礼开始了！），这与图片事实完全相反。

**step150-zh-case213** (准确性=2, 类型=predicate_error)
- 错句: 用户向“Mr Ako'o”索要其妻子的牧师联系方式，随后“Mr Ako'o”回复“Abbe Bindzi”并发送了消息。
- 说明: 动作主体颠倒且事实错误：是Mr Ako'o（左侧）发消息给右侧用户索要神父（Prêtre）的号码，以便和他妻子一起去见神父；右侧用户随后回复了Abbé Bindzi的联系方式。
- 完整理由: 摘要在核心事件的动作主体和对话逻辑上存在严重错误。首先，索要号码的是Mr Ako'o（左侧灰色气泡），而非摘要所说的“用户向Mr Ako'o索要”；其次，索要的是“神父（Prêtre）”的号码，而非“妻子的牧师”；最后，要求宣布结婚和给一箱啤酒的是右侧用户，而非Mr Ako'o。这些错误导致对话逻辑完全颠倒。

**step150-zh-case218** (准确性=2, 类型=entity_error)
- 错句: 利比亚多地正经历热浪，气温在48至50摄氏度之间波动，部分地区甚至达到43度。
- 说明: 图中文字说明气温普遍在43-48度，特定地区（阿齐齐亚和贾法拉）可达49-50度。摘要将43度描述为“甚至达到”，逻辑错误，且数值范围描述不精确。
- 完整理由: 摘要存在多处与图片事实不符的硬伤：1. 虚构了“埃及（尼罗河地区）”这一地点，图中文字仅提及利比亚的 العزيزية（阿齐齐亚）和 جفارة（贾法拉）；2. 错误归属了天气转折的时间，图中明确说是“周二（الثلاثاء）”下午开始降温，摘要写成了“周三”；3. 对气温数值的逻辑描述有误，图中称气温在43-48度之间，局部可达49-50度，摘要却称“甚至达到43度”，将低值误作极端高值；4. 凭空捏造了“早晚温差大，夜间依然炎热”的信息，图中并无此类描述。

</details>

### step300

#### en (English) — 15 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 21 | circumstantial_error | This image is a screenshot of a social media post, likely from TikTok, featuring a young woman wearing a brown hijab and a dark top. | 图片界面是 Instagram Reels 而非 TikTok，底部导航栏图标（如 Reels 图标、搜索、个人资料）是典型的 Instagram UI。 | 摘要存在多处事实性错误：首先，将界面误认为 TikTok，实际上是 Instagram Reels（底部导航栏图标和右侧互动栏布局均为 Instagram 特征）；其次，将文字语言误认为波斯语（Persian），实际上是库尔德语（Kurdish/Sorani）；最后，对核心文字的翻译不准确，漏掉了“拥抱（hug）”这一关键含义。 |
| 2 | 23 | entity_error | - **Text**: The text reads: "میں میں عورتیں لڑکیاں ہیں سب جاکر سب جاکر ... [重复内容]" | 摘要中给出的文字内容是完全虚构且重复的。图中实际文字为“محلے میں عورتیں لڑ رہی تھیں میں نے جا کر سمجھایا تب کہیں جا کر بات ہاتھا پائی تک پہنچی”，意为邻里妇女吵架，我去劝解结果打起来了。 | 摘要在识别界面元素和帖子来源方面是正确的，但在转录第一个帖子的文字内容时发生了严重的幻觉错误。它将原本简短的乌尔都语笑话替换成了一段极长且无意义的重复字符串，这与图片事实完全不符。 |
| 3 | 33 | entity_error | The current score is 109 out of a target of 5. | 109位于左上角，通常是关卡编号；5旁边有心形图标，代表生命值，而非目标分数。 | 摘要在游戏机制和数值归属上存在多处明确错误。首先，它将左上角的生命值（5颗心）误认为“目标分数”，并将109（可能是关卡号）误认为“当前分数”。其次，它将关卡目标（收集14个粉色龙宝宝）错误地描述为“特殊糖果的剩余使用次数”。此外，底部道具栏的图标描述不准确（如将刷子说成闪电、将棒棒糖锤说成宝石），且错误地声称每个道具都有剩余次数数字（实际上有两个显示的是“+”号）。 |
| 4 | 41 | entity_error | The post has garnered significant engagement, with 22K likes, 2.2K comments, and 70 shares. | 数值与归属错误：Nike 广告的点赞数是 222K 而非 22K；70 次分享属于上方的帖子，广告本身未显示分享数。 | 摘要在描述互动数据时存在明显的数值错误和归属错误。首先，Nike 广告的点赞数在图中明确显示为 '222K'，而摘要写成 '22K'，少了一个数量级；其次，'70' 次分享是属于广告上方那个帖子的数据，Nike 广告本身并未显示具体的分享数值，摘要将其错误归属于广告。这些属于 entity_error。 |
| 5 | 47 | entity_error | Event Scheduling: The group is coordinating to start an event at 8 PM. Firmansyah confirms the time and asks if everyone is ready. | 图中文字为“jam set 8”，在印尼语语境下指 7:30，而非 8 PM。 | 摘要在理解图内文字方面存在严重错误。首先，将 Ayu 发送的“Boles mas”（印尼语俚语，意为“可以/没问题”）误解为“身体不适（unwell）”，这属于严重的语义臆断错误。其次，将“jam set 8”（印尼语“jam setengah delapan”，即 7:30）误写为“8 PM”。由于存在明显的文字理解偏差和事实性错误，准确性评为 2 分。 |
| 6 | 60 | entity_error | The video has garnered significant engagement, with 14.6K likes, 462 comments, 3,284 shares, and 2,077 views. | 数值归属错误：3,284 对应的是书签图标（收藏/保存），而非分享（shares）；2,077 对应的是箭头图标（分享），而非播放量（views）。 | 摘要在描述社交媒体互动数据时存在明显的实体错误（entity_error）。它将书签图标对应的 3,284 误认为“分享数”（shares），实际上是收藏数；将分享箭头图标对应的 2,077 误认为“播放量”（views），实际上是分享数。根据评测标准，此类数值与图标归属的明确错误属于硬伤。 |
| 7 | 64 | entity_error | A notification from Safaricom indicates no active data bundle and suggests dialing *344# to purchase one. | 图中文字明确显示拨号代码为 *544#，而非 *344#。 | 摘要在处理文字细节方面存在多处严重的实体错误（entity_error）。首先，将拨号代码 *544# 错误记为 *344#；其次，多处人名拼写错误，如将 Sossygraze Onyango 写成 Sossyva Ong'anyo，将 ADHIAMBO 写成 ADILABO，将 Radong Muangi 写成 Radonyi Huangi；最后，将 Jumia 应用发出的通知错误归属于产品名 "Lemon"。这些错误涉及图中清晰可见的文字和数字，属于明显的实事性偏差。 |
| 8 | 66 | circumstantial_error | The image displays a screenshot of a social media post from the platform Instagram, featuring a video of a flooded street. | 图片界面显示的是 Facebook，而非 Instagram。 | 摘要存在明显的平台识别错误，将 Facebook 界面误认为 Instagram（图中包含 Facebook 特有的 Marketplace 图标、通知铃铛图标以及“was live”状态描述）。此外，在转录孟加拉语标题时存在拼写错误（多了一个元音符号）。虽然核心画面内容和互动数据描述准确，但平台归属错误属于实质性的情境错误。 |
| 9 | 71 | predicate_error | Message Flow: The conversation begins with the recipient sending a laughing emoji and the text “hahahhh” accompanied by a green frog emoji with hearts | 角色归属错误。在截图中，左侧粉色气泡是对方（whokilled.ley?）发送的，摘要将其称为“recipient”会导致逻辑混乱，通常应指明为对话的另一方或发送者。 | 摘要存在多处明显的硬错误：1. 凭空捏造了图中不存在的文字“PART 1 & 2”；2. 将视频会议界面中的“+19”（通常指参与人数）错误识别为“+19秒”的时间戳；3. 角色归属混乱，将左侧发送粉色气泡消息的用户（whokilled.ley?）称为“recipient（接收者）”，逻辑上存在矛盾。 |
| 10 | 93 | predicate_error | A person is shown carefully placing a screen protector onto the back of a smartphone. | 动作描述错误。画面显示是在手机正面屏幕上操作，而非背面（back）。 | 摘要在核心动作描述和数据归属上存在多处严重错误。首先，摘要称人正在将贴膜放在手机“背面（back）”，但画面明显是在手机屏幕（正面）上滴加液体；其次，摘要称正在“放置/对齐贴膜”，实际上图中手持的是小瓶子正在滴加液体（通常为UV胶）；最后，社交媒体互动数据归属错误：20对应的是分享图标，78对应的是收藏图标，摘要将其分别误认为评论数和分享数。 |
| 11 | 114 | entity_error | The image is a screenshot of a social media post promoting a Cambodian music video titled "Chhlu Meay" by the artist "Chhlu Meay." | 视频标题是“ចិត្តអើយ”（Chit Euy），而非“Chhlu Meay”。 | 摘要在核心实体识别上存在严重错误。它将视频标题和艺术家错误地识别为“Chhlu Meay”（ឆ្លុះមេយ），而图中文字清晰显示为“ចិត្តអើយ”（Chit Euy）。此外，它将“Subscribe”（ជាវ）按钮误认为“Follow”按钮。这些属于明显的实体错误（entity_error）。 |
| 12 | 156 | entity_error | Visual Elements: A hand holding a tarot card, stacked cards on a table, and a blue card labeled “FREEZING” visible in the foreground. | 前景中的蓝色卡片上写的文字是“FREEDOM”，而非“FREEZING”。 | 摘要存在两处明显的实体错误（entity_error）：一是将前景蓝色卡片上的文字“FREEDOM”误读为“FREEZING”；二是错误归属了社交媒体互动数据，图中显示评论数为32、分享数为7、收藏数为10，而摘要将其描述为点赞32、评论7、分享10。 |
| 13 | 186 | entity_error | A prominent overlay shows a TikTok video titled "Best Brazilian dance Brazil" by @camille..., which has garnered 1.1K likes and 2.3M views. | 视频作者应为“@Samugrimas-v6h”而非“@camille...”，点赞数应为“7.9K”而非“1.1K”。 | 摘要在描述视频叠加层（overlay）时存在明显的实体错误（entity_error）。它错误地将视频作者归属于“@camille...”，而图中清晰显示频道名为“@Samugrimas-v6h”；同时将点赞数误记为“1.1K”，而图中实际显示为“7.9K”。这些是图中可清晰辨认的文字和数值事实。 |
| 14 | 200 | entity_error | 2. **Password Setup**: Users are prompted to create a password, with a field to enter a 123-digit verification code. | 实体错误。图中“123”只是验证码输入框的占位图标或类型标识，摘要将其描述为“123位数字的验证码”是荒谬的错误。 | 摘要在关键数值上存在严重错误。首先，将验证码输入框的占位图标“123”误读为需要输入“123位数字”的验证码（123-digit verification code），这在逻辑上是不可能的，属于对UI元素的严重误解。其次，将广告文案中的奖金数额“৩৬৮”（孟加拉语数字368）错误地写成“768”，属于明显的实体错误。详见事实核查。 |
| 15 | 206 | predicate_error | The user asks about sales, to which Teresiah responds with "Bado sijafunga" (likely meaning "I haven't started yet" or similar in Swahili). | “Bado sijafunga”在商业语境下意为“我还没关门”，而非“还没开始”。 | 摘要在理解斯瓦希里语/Sheng语境方面存在多处严重错误，并出现了发送方归属错误。具体包括：将“Bado sijafunga”（还没关门）误译为“还没开始”；将关于酒桶状态的询问“Keg iko aje”误读为询问位置；将询问数量的“Daya uko na ngapi”误读为询问是否忙碌；将“Uko nusu”（剩一半）误读为“她很忙”；且错误地将 Teresiah 发送的消息“Tatu zinejaa”归于用户发送。 |

<details><summary>展开各 badcase 完整理由</summary>

**step300-en-case21** (准确性=2, 类型=circumstantial_error)
- 错句: This image is a screenshot of a social media post, likely from TikTok, featuring a young woman wearing a brown hijab and a dark top.
- 说明: 图片界面是 Instagram Reels 而非 TikTok，底部导航栏图标（如 Reels 图标、搜索、个人资料）是典型的 Instagram UI。
- 完整理由: 摘要存在多处事实性错误：首先，将界面误认为 TikTok，实际上是 Instagram Reels（底部导航栏图标和右侧互动栏布局均为 Instagram 特征）；其次，将文字语言误认为波斯语（Persian），实际上是库尔德语（Kurdish/Sorani）；最后，对核心文字的翻译不准确，漏掉了“拥抱（hug）”这一关键含义。

**step300-en-case23** (准确性=2, 类型=entity_error)
- 错句: - **Text**: The text reads: "میں میں عورتیں لڑکیاں ہیں سب جاکر سب جاکر ... [重复内容]"
- 说明: 摘要中给出的文字内容是完全虚构且重复的。图中实际文字为“محلے میں عورتیں لڑ رہی تھیں میں نے جا کر سمجھایا تب کہیں جا کر بات ہاتھا پائی تک پہنچی”，意为邻里妇女吵架，我去劝解结果打起来了。
- 完整理由: 摘要在识别界面元素和帖子来源方面是正确的，但在转录第一个帖子的文字内容时发生了严重的幻觉错误。它将原本简短的乌尔都语笑话替换成了一段极长且无意义的重复字符串，这与图片事实完全不符。

**step300-en-case33** (准确性=2, 类型=entity_error)
- 错句: The current score is 109 out of a target of 5.
- 说明: 109位于左上角，通常是关卡编号；5旁边有心形图标，代表生命值，而非目标分数。
- 完整理由: 摘要在游戏机制和数值归属上存在多处明确错误。首先，它将左上角的生命值（5颗心）误认为“目标分数”，并将109（可能是关卡号）误认为“当前分数”。其次，它将关卡目标（收集14个粉色龙宝宝）错误地描述为“特殊糖果的剩余使用次数”。此外，底部道具栏的图标描述不准确（如将刷子说成闪电、将棒棒糖锤说成宝石），且错误地声称每个道具都有剩余次数数字（实际上有两个显示的是“+”号）。

**step300-en-case41** (准确性=2, 类型=entity_error)
- 错句: The post has garnered significant engagement, with 22K likes, 2.2K comments, and 70 shares.
- 说明: 数值与归属错误：Nike 广告的点赞数是 222K 而非 22K；70 次分享属于上方的帖子，广告本身未显示分享数。
- 完整理由: 摘要在描述互动数据时存在明显的数值错误和归属错误。首先，Nike 广告的点赞数在图中明确显示为 '222K'，而摘要写成 '22K'，少了一个数量级；其次，'70' 次分享是属于广告上方那个帖子的数据，Nike 广告本身并未显示具体的分享数值，摘要将其错误归属于广告。这些属于 entity_error。

**step300-en-case47** (准确性=2, 类型=entity_error)
- 错句: Event Scheduling: The group is coordinating to start an event at 8 PM. Firmansyah confirms the time and asks if everyone is ready.
- 说明: 图中文字为“jam set 8”，在印尼语语境下指 7:30，而非 8 PM。
- 完整理由: 摘要在理解图内文字方面存在严重错误。首先，将 Ayu 发送的“Boles mas”（印尼语俚语，意为“可以/没问题”）误解为“身体不适（unwell）”，这属于严重的语义臆断错误。其次，将“jam set 8”（印尼语“jam setengah delapan”，即 7:30）误写为“8 PM”。由于存在明显的文字理解偏差和事实性错误，准确性评为 2 分。

**step300-en-case60** (准确性=2, 类型=entity_error)
- 错句: The video has garnered significant engagement, with 14.6K likes, 462 comments, 3,284 shares, and 2,077 views.
- 说明: 数值归属错误：3,284 对应的是书签图标（收藏/保存），而非分享（shares）；2,077 对应的是箭头图标（分享），而非播放量（views）。
- 完整理由: 摘要在描述社交媒体互动数据时存在明显的实体错误（entity_error）。它将书签图标对应的 3,284 误认为“分享数”（shares），实际上是收藏数；将分享箭头图标对应的 2,077 误认为“播放量”（views），实际上是分享数。根据评测标准，此类数值与图标归属的明确错误属于硬伤。

**step300-en-case64** (准确性=2, 类型=entity_error)
- 错句: A notification from Safaricom indicates no active data bundle and suggests dialing *344# to purchase one.
- 说明: 图中文字明确显示拨号代码为 *544#，而非 *344#。
- 完整理由: 摘要在处理文字细节方面存在多处严重的实体错误（entity_error）。首先，将拨号代码 *544# 错误记为 *344#；其次，多处人名拼写错误，如将 Sossygraze Onyango 写成 Sossyva Ong'anyo，将 ADHIAMBO 写成 ADILABO，将 Radong Muangi 写成 Radonyi Huangi；最后，将 Jumia 应用发出的通知错误归属于产品名 "Lemon"。这些错误涉及图中清晰可见的文字和数字，属于明显的实事性偏差。

**step300-en-case66** (准确性=2, 类型=circumstantial_error)
- 错句: The image displays a screenshot of a social media post from the platform Instagram, featuring a video of a flooded street.
- 说明: 图片界面显示的是 Facebook，而非 Instagram。
- 完整理由: 摘要存在明显的平台识别错误，将 Facebook 界面误认为 Instagram（图中包含 Facebook 特有的 Marketplace 图标、通知铃铛图标以及“was live”状态描述）。此外，在转录孟加拉语标题时存在拼写错误（多了一个元音符号）。虽然核心画面内容和互动数据描述准确，但平台归属错误属于实质性的情境错误。

**step300-en-case71** (准确性=2, 类型=predicate_error)
- 错句: Message Flow: The conversation begins with the recipient sending a laughing emoji and the text “hahahhh” accompanied by a green frog emoji with hearts.
- 说明: 角色归属错误。在截图中，左侧粉色气泡是对方（whokilled.ley?）发送的，摘要将其称为“recipient”会导致逻辑混乱，通常应指明为对话的另一方或发送者。
- 完整理由: 摘要存在多处明显的硬错误：1. 凭空捏造了图中不存在的文字“PART 1 & 2”；2. 将视频会议界面中的“+19”（通常指参与人数）错误识别为“+19秒”的时间戳；3. 角色归属混乱，将左侧发送粉色气泡消息的用户（whokilled.ley?）称为“recipient（接收者）”，逻辑上存在矛盾。

**step300-en-case93** (准确性=2, 类型=predicate_error)
- 错句: A person is shown carefully placing a screen protector onto the back of a smartphone.
- 说明: 动作描述错误。画面显示是在手机正面屏幕上操作，而非背面（back）。
- 完整理由: 摘要在核心动作描述和数据归属上存在多处严重错误。首先，摘要称人正在将贴膜放在手机“背面（back）”，但画面明显是在手机屏幕（正面）上滴加液体；其次，摘要称正在“放置/对齐贴膜”，实际上图中手持的是小瓶子正在滴加液体（通常为UV胶）；最后，社交媒体互动数据归属错误：20对应的是分享图标，78对应的是收藏图标，摘要将其分别误认为评论数和分享数。

**step300-en-case114** (准确性=2, 类型=entity_error)
- 错句: The image is a screenshot of a social media post promoting a Cambodian music video titled "Chhlu Meay" by the artist "Chhlu Meay."
- 说明: 视频标题是“ចិត្តអើយ”（Chit Euy），而非“Chhlu Meay”。
- 完整理由: 摘要在核心实体识别上存在严重错误。它将视频标题和艺术家错误地识别为“Chhlu Meay”（ឆ្លុះមេយ），而图中文字清晰显示为“ចិត្តអើយ”（Chit Euy）。此外，它将“Subscribe”（ជាវ）按钮误认为“Follow”按钮。这些属于明显的实体错误（entity_error）。

**step300-en-case156** (准确性=2, 类型=entity_error)
- 错句: Visual Elements: A hand holding a tarot card, stacked cards on a table, and a blue card labeled “FREEZING” visible in the foreground.
- 说明: 前景中的蓝色卡片上写的文字是“FREEDOM”，而非“FREEZING”。
- 完整理由: 摘要存在两处明显的实体错误（entity_error）：一是将前景蓝色卡片上的文字“FREEDOM”误读为“FREEZING”；二是错误归属了社交媒体互动数据，图中显示评论数为32、分享数为7、收藏数为10，而摘要将其描述为点赞32、评论7、分享10。

**step300-en-case186** (准确性=2, 类型=entity_error)
- 错句: A prominent overlay shows a TikTok video titled "Best Brazilian dance Brazil" by @camille..., which has garnered 1.1K likes and 2.3M views.
- 说明: 视频作者应为“@Samugrimas-v6h”而非“@camille...”，点赞数应为“7.9K”而非“1.1K”。
- 完整理由: 摘要在描述视频叠加层（overlay）时存在明显的实体错误（entity_error）。它错误地将视频作者归属于“@camille...”，而图中清晰显示频道名为“@Samugrimas-v6h”；同时将点赞数误记为“1.1K”，而图中实际显示为“7.9K”。这些是图中可清晰辨认的文字和数值事实。

**step300-en-case200** (准确性=2, 类型=entity_error)
- 错句: 2. **Password Setup**: Users are prompted to create a password, with a field to enter a 123-digit verification code.
- 说明: 实体错误。图中“123”只是验证码输入框的占位图标或类型标识，摘要将其描述为“123位数字的验证码”是荒谬的错误。
- 完整理由: 摘要在关键数值上存在严重错误。首先，将验证码输入框的占位图标“123”误读为需要输入“123位数字”的验证码（123-digit verification code），这在逻辑上是不可能的，属于对UI元素的严重误解。其次，将广告文案中的奖金数额“৩৬৮”（孟加拉语数字368）错误地写成“768”，属于明显的实体错误。详见事实核查。

**step300-en-case206** (准确性=2, 类型=predicate_error)
- 错句: The user asks about sales, to which Teresiah responds with "Bado sijafunga" (likely meaning "I haven't started yet" or similar in Swahili).
- 说明: “Bado sijafunga”在商业语境下意为“我还没关门”，而非“还没开始”。
- 完整理由: 摘要在理解斯瓦希里语/Sheng语境方面存在多处严重错误，并出现了发送方归属错误。具体包括：将“Bado sijafunga”（还没关门）误译为“还没开始”；将关于酒桶状态的询问“Keg iko aje”误读为询问位置；将询问数量的“Daya uko na ngapi”误读为询问是否忙碌；将“Uko nusu”（剩一半）误读为“她很忙”；且错误地将 Teresiah 发送的消息“Tatu zinejaa”归于用户发送。

</details>

#### fr (Français) — 20 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 18 | entity_error | La section « Chaînes » liste des canaux vidéo avec leurs noms, descriptions partielles, durées et nombre de vues ou d’interactions | 图中显示的数字是未读消息数，时间是最后一条消息的时间戳，而非视频时长或互动次数。 | 摘要在理解界面元素方面存在严重的系统性错误。它将 WhatsApp 频道列表中的消息时间戳（如 07:04, 22:52）误认为视频的“时长”（durée），并将绿色圆圈内的未读消息数（如 2, 55, 61）误认为“互动次数”或“观看次数”（interactions/vues）。此外，摘要称背景为“暗色”（sombre），而图片明显是浅色模式界面。这些属于明显的实体错误（entity_error）。 |
| 2 | 31 | circumstantial_error | L’image présente l’historique de visionnage d’un compte Instagram | 图片显示的是 TikTok 的界面，而非 Instagram。 | 摘要存在多处与图片事实明确矛盾的硬错误。首先，将界面错误识别为 Instagram，而实际上这是 TikTok 的观看历史界面（circumstantial_error）。其次，在数据处理上存在严重逻辑错误：声称 80.2K 是最高播放量，却在同一句中提到 3.8M 的视频，且在总结范围时称上限为 80.2K，完全忽略了图中清晰可见的 3.8M 和 3.1M 播放量（entity_error）。此外，还将墨镜错误地归属于播放量为 7.5K 的视频（entity_error）。 |
| 3 | 37 | entity_error | Une image transférée à 19:44 présente un compteur électrique monophasé 2 fils intelligent de la marque WASON, modèle DGBD101, certifié CEI 62053-21 et | 图中显示的认证标准为CEI 62053-21，并未提及62053-22。 | 摘要包含多处与图片事实明确矛盾的硬错误。首先，电表读数被错误地记为“640 kWh”，而图中清晰显示为“6.40”；其次，制造日期被写成“2026”，图中实际为“2020”；电流规格“5(80)A”被误写为“6/80A”；此外，摘要凭空捏造了图中不存在的文字“Wireless Communication”，并将“S”标志错误解读为无线通信（实际为STS标准标志）。 |
| 4 | 40 | entity_error | Documents officiels : Plusieurs images montrent des documents signés et estampillés, notamment un « FINANCEMENT PAR L’ÉTAT » et un « CERTIFICAT DE CON | 图中可见的文档文字是“FINANMOUGOU SABABOU GNOUMAN 1”和“TETCHI SAMSON”，摘要中提到的“FINANCEMENT PAR L’ÉTAT”和“CERTIFICAT DE CONSTRUCTION”属于凭空捏造的错误信息。 | 摘要在描述文档内容时出现了严重的幻觉（entity_error），将图中清晰可见的文字“FINANMOUGOU SABABOU GNOUMAN 1”和“TETCHI SAMSON”错误地表述为“FINANCEMENT PAR L’ÉTAT”和“CERTIFICAT DE CONSTRUCTION”，图中完全没有这些字样。此外，摘要称可见两段视频，但图中实际上有三段带有时间戳的视频（1:48, 0:11, 0:18）。 |
| 5 | 74 | entity_error | L’image montre une capture d’écran d’une conversation WhatsApp avec un contact nommé « Moilimatou Sanp... », contenant plusieurs vidéos transférées, d | 图中显示的是照片而非视频（无播放图标），且“00:28”是消息发送时间而非视频时长。 | 摘要存在多处明显的实体错误（entity_error）。首先，它将图片中的媒体文件误认为“视频”（vidéos），并把发送时间“00:28”误解为视频时长；在 WhatsApp 中，视频缩略图通常会有播放图标，而此处明显是照片。其次，摘要将背景文字“LA FORGE DES CHAMPIONS”错误地识别为“ATORGÉ DES CHAMPIONS”。 |
| 6 | 85 | predicate_error | Le destinataire a envoyé un message indiquant qu’il enverra quelque chose à la fin du mois, suivi d’un bonjour matinal (« Hello get up ») et d’une inv | 角色归属错误。这些消息是由左侧的联系人“Ettin”发送的，而非“接收者”（destinataire，即截图持有者）。 | 摘要在描述对话双方时存在严重的逻辑混乱和角色归属错误。它将左侧发送者（Ettin）发送的消息（如“Hello get up”、“Ok”）和右侧用户发送的消息（如“Yrs ett”）均归于“destinataire”（接收者），这在逻辑上是矛盾的，且与图片中明确的发送方标识不符。具体而言，它错误地声称接收者发送了“下月底发钱”的消息，并错误地声称接收者以“Ok”结束了对话（实际上“Ok”是左侧的 Ettin 发送的）。 |
| 7 | 106 | entity_error | L’image montre une conversation WhatsApp entre l’utilisateur et Suzanne, incluant un bordereau de versement espèces de la banque SCR Cameroon, un numé | 银行名称应为 SCB Cameroun，而非 SCR Cameroon。 | 摘要在关键细节上存在大量严重错误。首先，银行名称被错误识别为“SCR”而非“SCB”，并凭空捏造了银行全称；其次，单据上的所有关键信息（编号、客户名、账号、经办人姓名）均与图片文字严重不符；此外，摘要错误地将对方（Suzanne）发送的消息归属于用户；最后，摘要中提到的“102/102”页码属于凭空捏造的幻觉信息。 |
| 8 | 119 | entity_error | Certaines figures sont marquées « Buy » (ex. figure 1, 4, 7) | La figure 4 est explicitement marquée « Sell » en rouge sur le tableau, et non « Buy ». | Le résumé est globalement bien structuré mais contient des erreurs factuelles significatives concernant l'interprétation des données visuelles sur le tableau. Il attribue de manière erronée des signaux d'achat (Buy) à la figure 4 (qui est marquée 'Se |
| 9 | 120 | entity_error | Damso 226 : Commentaire publié il y a 20 heures, accompagné de trois emojis de visage souriant avec larmes de joie (😂😂😂) et d’un drapeau du Burkina Fa | 图片中 Damso 226 的评论是三个带爱心的笑脸（🥰🥰🥰），而非笑哭表情（😂😂😂）。 | 摘要在描述评论内容时存在多处明显的实体错误（entity_error）。它将用户 'Damso 226' 和 'EKOADE' 的表情符号错误地描述为 '😂😂😂'（笑哭），而图片中清晰显示为 '🥰🥰🥰'（带爱心的笑脸）。此外，它将用户 'reine championne' 用户名中的手指表情（👆）归类为评论内容的一部分，这也是不准确的。由于摘要的核心任务是详细列举评论内容，这些内容错误属于实质性偏差。 |
| 10 | 126 | entity_error | C : Dunlop — marque principalement associée au tennis et à la raquette, pas au golf. | 错误。Dunlop 也是非常著名的高尔夫品牌（Dunlop Sport），生产高尔夫球和球杆。 | 摘要在核心事实判断上出现了严重错误。它错误地声称 Dunlop 与高尔夫无关（实际上 Dunlop 是著名的高尔夫品牌），并虚构了 Kookaburra 生产高尔夫球杆的事实（实际上 Kookaburra 是板球和曲棍球品牌，通常被认为是该题的正确答案，即不属于高尔夫的品牌）。这种对图中实体属性的错误归属和虚构信息属于严重的 entity_error 和 out_of_context_error。 |
| 11 | 129 | entity_error | Ce contenu est une capture d’écran d’une vidéo TikTok diffusant un extrait de match de football entre le Bayern Munich et le Real Madrid, avec un focu | François Marchal 和 Christophe Jallet 是 Canal+ 的解说员/记者，而非球员。 | 摘要存在多处与图片事实明确矛盾的硬错误：1. 身份误判：将 François Marchal 和 Christophe Jallet（Canal+ 的解说员/记者）误认为球员；2. 比分/胜负误判：比分牌显示为 BAY 3-3 RMA (5\|4)，通常括号内数字对应前方队伍，即拜仁总比分或点球 5-4 领先，摘要却称皇马 5-4 领先；3. UI 细节错误：将塞内加尔国旗 🇸🇳 误认为瑞典国旗 🇸🇪，将山羊表情 🐐 误认为马 🐴；4. 场景臆断：在 88:25 这一常规赛时间点声称比赛已结束并 |
| 12 | 148 | entity_error | L’image présente deux actualités politiques et économiques de Côte d’Ivoire, publiées par koaci.com : une démission en cascade au sein du PPA-CI et la | 图片中的第一则新闻（PPA-CI 辞职）是由“LE DRONE”发布的，而非 koaci.com。只有第二则新闻来自 koaci.com。 | 摘要存在两处明显的实体错误：首先，它将两则新闻都归功于 koaci.com，但第一则新闻（关于 PPA-CI）实际上来自“LE DRONE”；其次，摘要声称总理和经济部长出席了奠基仪式，但奠基石上的文字显示出席者是 Ibrahim Kalil Konaté（数字经济部长）和 Bruno Nabagné Koné（建筑部长），并未提及总理。 |
| 13 | 149 | predicate_error | Le commandant adresse ses respects au président et lui souhaite bonne santé. | 错误。图中显示是省长（被用户称为总统）发消息说“Mes respects Commandant. Je vous espère en bonne santé”，即省长向指挥官致意，而非相反。 | 摘要在描述对话双方的身份和动作时存在严重的谓词错误（角色互换）。它系统性地将绿色气泡（用户/指挥官）发送的消息归功于“总统/省长”，并将灰色气泡（省长）发送的消息归功于“指挥官”。这种角色倒置误导了对话的实际流向。 |
| 14 | 177 | entity_error | Il s’agit d’une fiche d’inscription officielle pour un candidat au cycle de formation IDE-SSM-TSS, émise par le Ministère de la Santé de la République | 图中显示为“IDE-SFM-TSS”，摘要误写为“SSM”。 | 摘要中存在多处明显的实体错误（entity_error），涉及姓名、证件号、专业缩写及时间。例如：将姓氏“M'BOUAFFON”错误写为“Atoubaou”；将证件号开头的“CI”写为“CB”；将专业缩写“SFM”写为“SSM”，“IDE”写为“JDE”，“TBM”写为“ZEM”；将报名时间“00:38”写为“08h30”。这些错误严重影响了信息的真实性。 |
| 15 | 184 | entity_error | L’image présente une capture d’écran d’un post Facebook de la page « Wouri TV », accompagnée d’une liste de traductions françaises-anglaises et d’un m | 词汇表列表和Wouri TV的帖子是信息流中两个独立的帖子，而非同一个帖子相互伴随。 | 摘要存在多处明显的实体错误（entity_error）。首先，它错误地将21岁归属于母亲，而图中文字明确指出是“女儿21岁”（Ma fille de 21 ans）。其次，它将上方词汇表帖子的互动数据（1.7K点赞等）错误地归给下方的Wouri TV帖子。此外，它误称“15+”通知位于消息图标上，实际上是在首页和视频图标上。最后，它将Wouri TV帖子的432个赞误认为是评论或子帖子的点赞。 |
| 16 | 186 | entity_error | Le titre complet visible est « Berceuse pour Bébé ! Musique Douce pour u... », publié par @funKidsShow+9m, avec 867 vues et 6 clics sur « J’aime ». | 频道名称错误，图中为 @FunKidsShow-t9m，摘要写成了 +9m。 | 摘要在关键数值和实体名称上存在多处明确错误。首先，将 Shorts 视频的播放量“5 Md”（50亿，Milliard）和“1,6 Md”（16亿）错误地写成了“5 millions”（500万）和“1,6 million”（160万），数量级相差千倍，属于严重的 entity_error。其次，频道名称 @FunKidsShow-t9m 被错误写为 @funKidsShow+9m。尽管整体结构和场景描述正确，但这些实质性的数值错误导致准确性较低。 |
| 17 | 190 | predicate_error | Ce portrait présente un jeune enfant de profil, vêtu d’un ensemble coloré et coiffé de cheveux courts et bouclés. | 图中孩子是正对镜头的（de face），而非侧面（de profil）。 | 摘要存在多处与图片事实明确矛盾的硬错误：1. 摘要称孩子是“侧面（de profil）”，但图中孩子是正对镜头；2. 摘要在第4点称夹克是“关闭的（fermée）”，这不仅与第2点自相矛盾，也与图中夹克敞开的实情不符；3. 摘要将背景中的电线/绳索误认为“脸左侧的头发（mèche）”。 |
| 18 | 202 | entity_error | Vendredi 31 juillet : Messe funèbre à l’Église de l’Immaculée Conception (CHR), à partir de 12h30. | 图中显示周五20h在Dimbokro举行守灵（Veillée funèbre）。摘要中的“12h30”是误读了社交媒体界面的发布时间“12:39”，且教堂名称属于虚构。 | 摘要在核心的葬礼日程信息上存在多处严重的事实性错误和虚构。它将社交媒体界面的发布时间（12:39）误认为仪式时间（12h30），将“遗体告别（Levée de corps）”的日期从周六（01 AOÛT）改成了周日（2 août），并凭空捏造了图中未出现的教堂名称（Église de l’Immaculée Conception）和墓地信息。尽管整体框架符合讣告格式，但关键数据与图片事实严重不符。 |
| 19 | 211 | out_of_context_error | L’image montre les éléments typiques d’une vidéo TikTok, y compris les boutons d’interaction (like, commentaire, partage)... | 图中并未显示点赞（心形）和评论（气泡）图标，仅显示了收藏和分享图标。 | 摘要在描述社交媒体互动数据时存在多处严重的事实性错误。它将书签图标旁的数字“99”（收藏数）错误地识别为“观看次数”（compteur de vues），将分享图标旁的数字“33”（分享数）错误地识别为“消息数量”（nombre de messages）。此外，摘要声称图中显示了点赞和评论按钮，但这些图标在提供的截图中并不可见。 |
| 20 | 219 | entity_error | Le compte qui a publié le Reel est « Cheick M Diarra », identifié par un avatar circulaire et un badge de vérification. | 图中名字旁边并没有蓝色验证徽章（badge de vérification）。 | 摘要包含多处明确的事实错误：1. 虚构了不存在的“验证徽章”（badge de vérification）；2. 错误归属了互动数据，将分享数（27）误认为评论数，且图中评论图标旁并无数字；3. 错误识别了国家代码，+223 是马里（Mali）的代码，而非布基纳法索（Burkina Faso，+226）。 |

<details><summary>展开各 badcase 完整理由</summary>

**step300-fr-case18** (准确性=2, 类型=entity_error)
- 错句: La section « Chaînes » liste des canaux vidéo avec leurs noms, descriptions partielles, durées et nombre de vues ou d’interactions
- 说明: 图中显示的数字是未读消息数，时间是最后一条消息的时间戳，而非视频时长或互动次数。
- 完整理由: 摘要在理解界面元素方面存在严重的系统性错误。它将 WhatsApp 频道列表中的消息时间戳（如 07:04, 22:52）误认为视频的“时长”（durée），并将绿色圆圈内的未读消息数（如 2, 55, 61）误认为“互动次数”或“观看次数”（interactions/vues）。此外，摘要称背景为“暗色”（sombre），而图片明显是浅色模式界面。这些属于明显的实体错误（entity_error）。

**step300-fr-case31** (准确性=2, 类型=circumstantial_error)
- 错句: L’image présente l’historique de visionnage d’un compte Instagram
- 说明: 图片显示的是 TikTok 的界面，而非 Instagram。
- 完整理由: 摘要存在多处与图片事实明确矛盾的硬错误。首先，将界面错误识别为 Instagram，而实际上这是 TikTok 的观看历史界面（circumstantial_error）。其次，在数据处理上存在严重逻辑错误：声称 80.2K 是最高播放量，却在同一句中提到 3.8M 的视频，且在总结范围时称上限为 80.2K，完全忽略了图中清晰可见的 3.8M 和 3.1M 播放量（entity_error）。此外，还将墨镜错误地归属于播放量为 7.5K 的视频（entity_error）。

**step300-fr-case37** (准确性=2, 类型=entity_error)
- 错句: Une image transférée à 19:44 présente un compteur électrique monophasé 2 fils intelligent de la marque WASON, modèle DGBD101, certifié CEI 62053-21 et 62053-22.
- 说明: 图中显示的认证标准为CEI 62053-21，并未提及62053-22。
- 完整理由: 摘要包含多处与图片事实明确矛盾的硬错误。首先，电表读数被错误地记为“640 kWh”，而图中清晰显示为“6.40”；其次，制造日期被写成“2026”，图中实际为“2020”；电流规格“5(80)A”被误写为“6/80A”；此外，摘要凭空捏造了图中不存在的文字“Wireless Communication”，并将“S”标志错误解读为无线通信（实际为STS标准标志）。

**step300-fr-case40** (准确性=2, 类型=entity_error)
- 错句: Documents officiels : Plusieurs images montrent des documents signés et estampillés, notamment un « FINANCEMENT PAR L’ÉTAT » et un « CERTIFICAT DE CONSTRUCTION », indiquant une démarche administrative ou légale liée à un bien immobilier.
- 说明: 图中可见的文档文字是“FINANMOUGOU SABABOU GNOUMAN 1”和“TETCHI SAMSON”，摘要中提到的“FINANCEMENT PAR L’ÉTAT”和“CERTIFICAT DE CONSTRUCTION”属于凭空捏造的错误信息。
- 完整理由: 摘要在描述文档内容时出现了严重的幻觉（entity_error），将图中清晰可见的文字“FINANMOUGOU SABABOU GNOUMAN 1”和“TETCHI SAMSON”错误地表述为“FINANCEMENT PAR L’ÉTAT”和“CERTIFICAT DE CONSTRUCTION”，图中完全没有这些字样。此外，摘要称可见两段视频，但图中实际上有三段带有时间戳的视频（1:48, 0:11, 0:18）。

**step300-fr-case74** (准确性=2, 类型=entity_error)
- 错句: L’image montre une capture d’écran d’une conversation WhatsApp avec un contact nommé « Moilimatou Sanp... », contenant plusieurs vidéos transférées, dont certaines sont marquées comme « Transféré » et accompagnées de durées (00:28) et de coches de lecture.
- 说明: 图中显示的是照片而非视频（无播放图标），且“00:28”是消息发送时间而非视频时长。
- 完整理由: 摘要存在多处明显的实体错误（entity_error）。首先，它将图片中的媒体文件误认为“视频”（vidéos），并把发送时间“00:28”误解为视频时长；在 WhatsApp 中，视频缩略图通常会有播放图标，而此处明显是照片。其次，摘要将背景文字“LA FORGE DES CHAMPIONS”错误地识别为“ATORGÉ DES CHAMPIONS”。

**step300-fr-case85** (准确性=2, 类型=predicate_error)
- 错句: Le destinataire a envoyé un message indiquant qu’il enverra quelque chose à la fin du mois, suivi d’un bonjour matinal (« Hello get up ») et d’une invitation à se lever et lire (« Get up and read »).
- 说明: 角色归属错误。这些消息是由左侧的联系人“Ettin”发送的，而非“接收者”（destinataire，即截图持有者）。
- 完整理由: 摘要在描述对话双方时存在严重的逻辑混乱和角色归属错误。它将左侧发送者（Ettin）发送的消息（如“Hello get up”、“Ok”）和右侧用户发送的消息（如“Yrs ett”）均归于“destinataire”（接收者），这在逻辑上是矛盾的，且与图片中明确的发送方标识不符。具体而言，它错误地声称接收者发送了“下月底发钱”的消息，并错误地声称接收者以“Ok”结束了对话（实际上“Ok”是左侧的 Ettin 发送的）。

**step300-fr-case106** (准确性=2, 类型=entity_error)
- 错句: L’image montre une conversation WhatsApp entre l’utilisateur et Suzanne, incluant un bordereau de versement espèces de la banque SCR Cameroon, un numéro de téléphone transféré, un appel vocal et un message textuel.
- 说明: 银行名称应为 SCB Cameroun，而非 SCR Cameroon。
- 完整理由: 摘要在关键细节上存在大量严重错误。首先，银行名称被错误识别为“SCR”而非“SCB”，并凭空捏造了银行全称；其次，单据上的所有关键信息（编号、客户名、账号、经办人姓名）均与图片文字严重不符；此外，摘要错误地将对方（Suzanne）发送的消息归属于用户；最后，摘要中提到的“102/102”页码属于凭空捏造的幻觉信息。

**step300-fr-case119** (准确性=2, 类型=entity_error)
- 错句: Certaines figures sont marquées « Buy » (ex. figure 1, 4, 7)
- 说明: La figure 4 est explicitement marquée « Sell » en rouge sur le tableau, et non « Buy ».
- 完整理由: Le résumé est globalement bien structuré mais contient des erreurs factuelles significatives concernant l'interprétation des données visuelles sur le tableau. Il attribue de manière erronée des signaux d'achat (Buy) à la figure 4 (qui est marquée 'Sell') et des signaux de vente (Sell) aux figures 5 et 9 (qui indiquent des signaux d'achat en vert). Ces erreurs d'entité (entity_error) contredisent directement le contenu textuel et visuel du tableau blanc.

**step300-fr-case120** (准确性=2, 类型=entity_error)
- 错句: Damso 226 : Commentaire publié il y a 20 heures, accompagné de trois emojis de visage souriant avec larmes de joie (😂😂😂) et d’un drapeau du Burkina Faso, indiquant un ton humoristique ou patriotique.
- 说明: 图片中 Damso 226 的评论是三个带爱心的笑脸（🥰🥰🥰），而非笑哭表情（😂😂😂）。
- 完整理由: 摘要在描述评论内容时存在多处明显的实体错误（entity_error）。它将用户 'Damso 226' 和 'EKOADE' 的表情符号错误地描述为 '😂😂😂'（笑哭），而图片中清晰显示为 '🥰🥰🥰'（带爱心的笑脸）。此外，它将用户 'reine championne' 用户名中的手指表情（👆）归类为评论内容的一部分，这也是不准确的。由于摘要的核心任务是详细列举评论内容，这些内容错误属于实质性偏差。

**step300-fr-case126** (准确性=2, 类型=entity_error)
- 错句: C : Dunlop — marque principalement associée au tennis et à la raquette, pas au golf.
- 说明: 错误。Dunlop 也是非常著名的高尔夫品牌（Dunlop Sport），生产高尔夫球和球杆。
- 完整理由: 摘要在核心事实判断上出现了严重错误。它错误地声称 Dunlop 与高尔夫无关（实际上 Dunlop 是著名的高尔夫品牌），并虚构了 Kookaburra 生产高尔夫球杆的事实（实际上 Kookaburra 是板球和曲棍球品牌，通常被认为是该题的正确答案，即不属于高尔夫的品牌）。这种对图中实体属性的错误归属和虚构信息属于严重的 entity_error 和 out_of_context_error。

**step300-fr-case129** (准确性=2, 类型=entity_error)
- 错句: Ce contenu est une capture d’écran d’une vidéo TikTok diffusant un extrait de match de football entre le Bayern Munich et le Real Madrid, avec un focus sur les joueurs François Marchal et Christophe Jallet.
- 说明: François Marchal 和 Christophe Jallet 是 Canal+ 的解说员/记者，而非球员。
- 完整理由: 摘要存在多处与图片事实明确矛盾的硬错误：1. 身份误判：将 François Marchal 和 Christophe Jallet（Canal+ 的解说员/记者）误认为球员；2. 比分/胜负误判：比分牌显示为 BAY 3-3 RMA (5|4)，通常括号内数字对应前方队伍，即拜仁总比分或点球 5-4 领先，摘要却称皇马 5-4 领先；3. UI 细节错误：将塞内加尔国旗 🇸🇳 误认为瑞典国旗 🇸🇪，将山羊表情 🐐 误认为马 🐴；4. 场景臆断：在 88:25 这一常规赛时间点声称比赛已结束并进入点球大战，且凭空捏造了“球员退役”的背景。

**step300-fr-case148** (准确性=2, 类型=entity_error)
- 错句: L’image présente deux actualités politiques et économiques de Côte d’Ivoire, publiées par koaci.com : une démission en cascade au sein du PPA-CI et la pose de la première pierre d’un projet de transformation du cacao à San Pedro.
- 说明: 图片中的第一则新闻（PPA-CI 辞职）是由“LE DRONE”发布的，而非 koaci.com。只有第二则新闻来自 koaci.com。
- 完整理由: 摘要存在两处明显的实体错误：首先，它将两则新闻都归功于 koaci.com，但第一则新闻（关于 PPA-CI）实际上来自“LE DRONE”；其次，摘要声称总理和经济部长出席了奠基仪式，但奠基石上的文字显示出席者是 Ibrahim Kalil Konaté（数字经济部长）和 Bruno Nabagné Koné（建筑部长），并未提及总理。

**step300-fr-case149** (准确性=2, 类型=predicate_error)
- 错句: Le commandant adresse ses respects au président et lui souhaite bonne santé.
- 说明: 错误。图中显示是省长（被用户称为总统）发消息说“Mes respects Commandant. Je vous espère en bonne santé”，即省长向指挥官致意，而非相反。
- 完整理由: 摘要在描述对话双方的身份和动作时存在严重的谓词错误（角色互换）。它系统性地将绿色气泡（用户/指挥官）发送的消息归功于“总统/省长”，并将灰色气泡（省长）发送的消息归功于“指挥官”。这种角色倒置误导了对话的实际流向。

**step300-fr-case177** (准确性=2, 类型=entity_error)
- 错句: Il s’agit d’une fiche d’inscription officielle pour un candidat au cycle de formation IDE-SSM-TSS, émise par le Ministère de la Santé de la République de Côte d’Ivoire.
- 说明: 图中显示为“IDE-SFM-TSS”，摘要误写为“SSM”。
- 完整理由: 摘要中存在多处明显的实体错误（entity_error），涉及姓名、证件号、专业缩写及时间。例如：将姓氏“M'BOUAFFON”错误写为“Atoubaou”；将证件号开头的“CI”写为“CB”；将专业缩写“SFM”写为“SSM”，“IDE”写为“JDE”，“TBM”写为“ZEM”；将报名时间“00:38”写为“08h30”。这些错误严重影响了信息的真实性。

**step300-fr-case184** (准确性=2, 类型=entity_error)
- 错句: L’image présente une capture d’écran d’un post Facebook de la page « Wouri TV », accompagnée d’une liste de traductions françaises-anglaises et d’un message viral partagé par une mère.
- 说明: 词汇表列表和Wouri TV的帖子是信息流中两个独立的帖子，而非同一个帖子相互伴随。
- 完整理由: 摘要存在多处明显的实体错误（entity_error）。首先，它错误地将21岁归属于母亲，而图中文字明确指出是“女儿21岁”（Ma fille de 21 ans）。其次，它将上方词汇表帖子的互动数据（1.7K点赞等）错误地归给下方的Wouri TV帖子。此外，它误称“15+”通知位于消息图标上，实际上是在首页和视频图标上。最后，它将Wouri TV帖子的432个赞误认为是评论或子帖子的点赞。

**step300-fr-case186** (准确性=2, 类型=entity_error)
- 错句: Le titre complet visible est « Berceuse pour Bébé ! Musique Douce pour u... », publié par @funKidsShow+9m, avec 867 vues et 6 clics sur « J’aime ».
- 说明: 频道名称错误，图中为 @FunKidsShow-t9m，摘要写成了 +9m。
- 完整理由: 摘要在关键数值和实体名称上存在多处明确错误。首先，将 Shorts 视频的播放量“5 Md”（50亿，Milliard）和“1,6 Md”（16亿）错误地写成了“5 millions”（500万）和“1,6 million”（160万），数量级相差千倍，属于严重的 entity_error。其次，频道名称 @FunKidsShow-t9m 被错误写为 @funKidsShow+9m。尽管整体结构和场景描述正确，但这些实质性的数值错误导致准确性较低。

**step300-fr-case190** (准确性=2, 类型=predicate_error)
- 错句: Ce portrait présente un jeune enfant de profil, vêtu d’un ensemble coloré et coiffé de cheveux courts et bouclés.
- 说明: 图中孩子是正对镜头的（de face），而非侧面（de profil）。
- 完整理由: 摘要存在多处与图片事实明确矛盾的硬错误：1. 摘要称孩子是“侧面（de profil）”，但图中孩子是正对镜头；2. 摘要在第4点称夹克是“关闭的（fermée）”，这不仅与第2点自相矛盾，也与图中夹克敞开的实情不符；3. 摘要将背景中的电线/绳索误认为“脸左侧的头发（mèche）”。

**step300-fr-case202** (准确性=2, 类型=entity_error)
- 错句: Vendredi 31 juillet : Messe funèbre à l’Église de l’Immaculée Conception (CHR), à partir de 12h30.
- 说明: 图中显示周五20h在Dimbokro举行守灵（Veillée funèbre）。摘要中的“12h30”是误读了社交媒体界面的发布时间“12:39”，且教堂名称属于虚构。
- 完整理由: 摘要在核心的葬礼日程信息上存在多处严重的事实性错误和虚构。它将社交媒体界面的发布时间（12:39）误认为仪式时间（12h30），将“遗体告别（Levée de corps）”的日期从周六（01 AOÛT）改成了周日（2 août），并凭空捏造了图中未出现的教堂名称（Église de l’Immaculée Conception）和墓地信息。尽管整体框架符合讣告格式，但关键数据与图片事实严重不符。

**step300-fr-case211** (准确性=2, 类型=out_of_context_error)
- 错句: L’image montre les éléments typiques d’une vidéo TikTok, y compris les boutons d’interaction (like, commentaire, partage)...
- 说明: 图中并未显示点赞（心形）和评论（气泡）图标，仅显示了收藏和分享图标。
- 完整理由: 摘要在描述社交媒体互动数据时存在多处严重的事实性错误。它将书签图标旁的数字“99”（收藏数）错误地识别为“观看次数”（compteur de vues），将分享图标旁的数字“33”（分享数）错误地识别为“消息数量”（nombre de messages）。此外，摘要声称图中显示了点赞和评论按钮，但这些图标在提供的截图中并不可见。

**step300-fr-case219** (准确性=2, 类型=entity_error)
- 错句: Le compte qui a publié le Reel est « Cheick M Diarra », identifié par un avatar circulaire et un badge de vérification.
- 说明: 图中名字旁边并没有蓝色验证徽章（badge de vérification）。
- 完整理由: 摘要包含多处明确的事实错误：1. 虚构了不存在的“验证徽章”（badge de vérification）；2. 错误归属了互动数据，将分享数（27）误认为评论数，且图中评论图标旁并无数字；3. 错误识别了国家代码，+223 是马里（Mali）的代码，而非布基纳法索（Burkina Faso，+226）。

</details>

#### ru (Русский) — 19 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 7 | entity_error | Детский тариф (от 1 метра до 12 лет) Действует в будние дни: 1500 ₽ за день. В выходные: 800 ₽ за день. | 错误。图中1500 ₽是成人周末/节假日票价，800 ₽是儿童周末/节假日票价。摘要将两者混淆并错误归类为儿童票价。 | 摘要在前半部分描述准确，但在第4点关于价格的描述中出现了严重的实体错误（将成人周末票价1500元误认为儿童票价，且逻辑混乱）。此外，最后一段出现了严重的重复错误（looping），导致信息完全失真。 |
| 2 | 13 | entity_error | Справа от видео расположены кнопки лайка (40), комментариев (459), репоста (262) и меню (три точки). | 数值与图标对应错误：图中气泡（评论）为40，双箭头（Remix）为459，单箭头（分享）为262，心形（点赞）无数字。摘要完全搞混了这些数值的归属。 | 摘要在描述互动数据时存在明显的实体错误（entity_error）。它将点赞数误报为40（图中点赞图标旁无数字），将评论数误报为459（图中气泡图标旁为40），将转发数误报为262（图中双箭头图标旁为459，单箭头分享图标旁为262）。这种数值与图标的错误挂钩属于严重的虚构与错误归属。 |
| 3 | 16 | entity_error | Сзади автомобиля стоит женщина с поднятыми руками, выражая радость или удивление. | На фото изображен мужчина с длинными волосами, который стоит в открытом люке автомобиля, а не женщина позади него. | В аннотации допущены серьезные фактические ошибки: неверно определен пол и местоположение человека (мужчина в люке, а не женщина сзади), а также неверно указано количество лайков (1,6 млн вместо 1,3 млн). См. пункты 3 и 7 в проверке фактов. |
| 4 | 50 | entity_error | 1. **Действие с тормозом**: Рука человека нажимает на рычаг дискового тормоза, расположенного на вилке велосипеда. | На фото рука касается тормозного суппорта (caliper), а не рычага (lever). Кроме того, тормоз находится на заднем перье рамы, а не на вилке. | В аннотации содержатся существенные фактические ошибки, противоречащие изображению. Во-первых, на тормозном суппорте четко видна надпись «SRAM», а не «SHIMANO». Во-вторых, рука касается самого тормозного механизма (суппорта), а не рычага (ручки) торм |
| 5 | 73 | entity_error | Устройство отображает время 21:57, дату 27.07, температуру 26,4°C и влажность 7,27%. | 存在两处错误：1. 图中底部的“7.27.”显然是日期（7月27日），摘要将其误认为湿度“7,27%”；2. 图中右上角显示数字“26”，摘要凭空添加了小数位描述为“26,4°C”。 | 摘要在描述数字信息时存在明显的实体错误（entity_error）。它将日期“7.27.”错误地识别为湿度“7,27%”，并且在温度数值中凭空捏造了小数点后的数字“26,4°C”（图中仅可见“26”）。 |
| 6 | 86 | entity_error | Название видео: «Прожил Пятую Ночь с ПИВОЗАВРОМ в ПОДЪЕЗДЕ! (5 НО...» — указывает на серию контента, связанную с выживанием и юмористическим элементом | 该标题属于下方推荐的视频，主视频标题应为“Прожил Пять Ночей с ПИВОЗАВРОМ в...”。 | 摘要存在多处严重的实质性错误：1. 数量级错误：将 538 个“不喜欢”误报为 538,000（538 тыс.），夸大了三个数量级；2. 实体归属错误：将下方推荐视频的标题误认为主视频标题，并将该视频的播放量（358k）误认为是整个频道的播放量；3. 包含大量主观臆断和元描述（对摘要自身的评价），且结尾句子残缺。 |
| 7 | 96 | entity_error | На экране видны элементы управления: таймер (12), счётчик очков (50/50), кнопки «10», «10», «10», «50». | На кнопках в игре написаны другие значения: 50/50, 20, значок обновления, 10, значок перемотки, 50. | Хотя общая структура и основные разделы скриншота определены верно, в деталях содержатся грубые фактические ошибки. В разделе игры названия песен и исполнителей сильно искажены (галлюцинации текста): «Как утонули в России» вместо «Как упоительны в Ро |
| 8 | 97 | entity_error | Диалог ведётся в мессенджере, где пользователь делится личными воспоминаниями о школьных годах и необычном опыте танцев в лагере, а Антон Максимов отв | На скриншоте видна только одна ответная реплика Антона («Пойдем куда нибудь потанцем»), использование множественного числа («репликами») не совсем точно. | Основная ошибка заключается в неверном определении отправителя ключевого сообщения в конце диалога. Сообщение «Пойдем куда нибудь потанцем» находится в сером облачке слева, что в интерфейсе Telegram однозначно указывает на ответ собеседника (Антона М |
| 9 | 104 | grammatical_error | На изображении запечатлены два человека за столом,举杯 в бокалах с красным вином, рядом с ними — блюдо из мяса с луком. | 句子中夹杂了中文词汇“举杯”，属于严重的语法/语种错误；同时存在 entity_error，因为右侧的人拿的不是红酒杯。 | 摘要存在明显的语言混合错误（在俄语中夹杂了中文“举杯”），且存在事实性错误：摘要声称两人都拿着装有红酒的高脚杯，但实际上右侧戴面具的人拿着的是一个装有棕色液体的小玻璃杯（烈酒杯），并非红酒高脚杯。 |
| 10 | 110 | predicate_error | Видео демонстрирует мужчину в кепке и полосатой футболке, указывающего на объект — вероятно, устройство или контейнер с бумагой... | 图中男子位于左侧，而指向物体的动作是由右侧画面中的手完成的，这是合拍视频，动作主体归属错误。 | 摘要存在两处明显的硬伤：一是动作归属错误（predicate_error），将右侧画面中手指指向物体的动作归于左侧的男子，实际上这是双人合拍/反应类视频，动作主体并非该男子；二是实体属性错误（entity_error），将纸上明显的韩文（Hangeul）误认为英文。 |
| 11 | 122 | entity_error | Изображение представляет собой скриншот из социальной сети, где пользователь shahrom_journalist опубликовал сообщение с изображением экрана смартфона, | 聊天消息的语言是塔吉克语，而非俄语（仅系统提示语是俄语）。 | 摘要在描述核心内容（聊天记录）时存在大量严重的事实性错误。它错误地将聊天语言识别为俄语（实际为塔吉克语），错误地转录了几乎所有的时间（如将1:43写成14:49，将08:30写成00:20）、金额（将1500$写成16008）和背景数字（将-436写成-435）。此外，它还将消息内容误认为发送者姓名，并对最后一条关键消息进行了严重的幻觉式转录。 |
| 12 | 126 | entity_error | Изображение представляет собой страницы из книги А. Корнева «Магия. Сакральные обряды и ритуалы. Практическое руководство» | 书名错误，图中显示为“Сильнейшие обряды и ритуалы”，而非“Сакральные”。 | 摘要存在多处严重的事实性错误和幻觉。首先，书名被错误引用（图中为“Сильнейшие”，摘要写成“Сакральные”）。其次，在描述第一个仪式时，将“大米”（рис）误认为“格栅”（решётка），并将“硬币排列成梯形”误描述为“不同面值的纸币”。在第二个仪式的背景故事中，摘要完全颠倒了人物关系：图中是熟人向作者求助，摘要却写成熟人帮助了作者。此外，摘要还虚构了书中不存在的引用语（“祈祷吧，一切都会有的”）和建议（“不要立即检查结果”）。 |
| 13 | 131 | out_of_context_error | Решение проблем с недвижимостью, земельными спорами, наследственными делами. | 这是凭空捏造的信息。原文第一段明确提到的是“逾期或有问题的贷款”（seskhi），而非房地产、土地或遗产。 | 摘要存在严重的幻觉和事实错误。首先，它完全遗漏了信息的核心主题——针对“逾期或有问题的贷款”（格鲁吉亚语：seskhi）以及收债公司的骚扰提供法律援助，反而凭空捏造了“房地产、土地纠纷、遗产”等文中未提及的业务领域。其次，摘要声称文中包含“法院”和“文件”等关键词，但格鲁吉亚语原文中并未出现对应的词汇（sasamarTlo 或 sabutebi）。 |
| 14 | 168 | entity_error | Автор выражает сильное разочарование и стресс после сдачи зачета по биологии, используя фразы вроде «Истерика въебала» и «Хуй ег о знает». | 图中文字为“бх”，通常指生物化学（биохимия），而非生物学（биология）。 | 摘要存在多处实质性事实错误：1. 将“бх”（生物化学）错误识别为“биология”（生物学）；2. 将“общежитие”（宿舍）错误表述为“учебное заведение”（教育机构）；3. 最严重的错误是颠倒了帮助关系，图中文字明确表示作者帮助了朋友们（помогла сдать），而摘要却说作者感谢他们的帮助；4. 摘要中还存在大段内容的重复（见事实核查第11、12条）。 |
| 15 | 177 | entity_error | Некоторые игроки помечены как травмированные (например, Даниэль Ким, Даниэль Матей), другие — в процессе восстановления (например, Мессиах, Паскаль Го | 图中出现的受伤球员是 Гриз Ли 和 Погрыз Дикаприо。摘要中提到的所有姓名均为凭空捏造，图中不存在。 | 摘要包含严重的虚构和技术性错误。首先，摘要中提到的球员姓名（如 Даниэль Ким, Даниэль Матей, Мессиах, Паскаль Гомес）在图片中完全不存在，属于凭空捏造（entity_error）。其次，摘要列出的位置缩写（如 ЛК, ЦК, РК, ЛБ, ЦБ, РБ 等）大部分未在图中出现，图中仅可见 Gk, Ld, Cd, Rd, Lm, Cm, Rm, Cf, Rf。最严重的是，摘要进入了无限重复的循环并被截断，这属于严重的生成故障。 |
| 16 | 183 | entity_error | Один из участников, одетый в белую майку с надписью «SUNRISE» и номером «1», находится впереди, двое других — позади него, готовые к действию. | 图中右侧男子的背心文字是“RUSSIAN BEAR”，而非“SUNRISE”，且无数字“1”；三人基本并排，并非一前两后。 | 摘要在描述主体细节时存在明显的虚构和错误归属。摘要声称其中一名参与者穿着印有“SUNRISE”字样和数字“1”的白色背心，但图片中右侧男子的灰色背心上清晰可见的文字是“RUSSIAN BEAR”，且没有任何数字“1”。这种凭空捏造文字信息的行为属于严重的 entity_error。此外，摘要描述参与者的位置为“一前两后”，而图中三人基本并排站在轮胎后方。 |
| 17 | 197 | entity_error | Документ является официальным уведомлением от комиссии по делам несовершеннолетних (КДН) о поступлении протокола в отношении гражданина по факту нахож | 文中括号内是占位符“(ФИО)”（姓名缩写），而非“РГО”。 | 摘要存在多处严重的事实性错误。首先，将占位符“(ФИО)”（姓名缩写）误读为“РГО”；其次，将地址“ул. Горького 14а каб. 22”错误写成“ул. Горького, д. 22, кв. 22”；再次，将“不出席的后果（в случае неявки）”误解为“再次违规的后果（при повторном нарушении）”；最后，将法律术语“привод”（强制传唤/拘留）误写为“приём”（接待）。此外，摘要还包含大量文中未提及的通用建议。 |
| 18 | 208 | entity_error | Упоминает ответ Тегерана, заявление СБМГИ и геополитические последствия. | 图中文字为“СИБИГИ”（指代政治人物 Сибига），摘要误写为“СБМГИ”。 | 摘要存在多处实质性事实错误：1. 错误归属数值，将第一条视频的分享数“9”误认为第二条视频的评论数；2. 错误扩大范围，声称“两部视频”均标有“18+”和外国代理人提示，但图中仅第一部视频有此提示；3. 文字识别错误，将缩略图中的“СИБИГИ”写成“СБМГИ”；4. 包含大量脱离语境的臆断和建议（如“用户建议”部分），且结尾文字残缺。 |
| 19 | 211 | entity_error | Левый мужчина носит тактический жилет коричневого цвета поверх рубашки, солнцезащитные очки на голове и имеет короткую стрижку. | 左侧男子穿的是一件躯干部分为棕色的作战蛙服（Combat Shirt），而不是在衬衫外面套了战术背心。 | 摘要存在严重的实体错误。首先，右侧男子帽子上的文字清晰可见为“Я НЕ ГРУСТНЫЙ, Я ТРЕЗВЫЙ”（我不难过，我很清醒），而摘要将其错误地识别为“ВООРУЖЕННЫЕ СИЛЫ УКРАИНЫ”（乌克兰武装部队），这属于严重的虚构和错误归属。其次，左侧男子穿着的是躯干为棕色的作战服（Combat Shirt），而非摘要所称的“棕色战术背心”。 |

<details><summary>展开各 badcase 完整理由</summary>

**step300-ru-case7** (准确性=2, 类型=entity_error)
- 错句: Детский тариф (от 1 метра до 12 лет) Действует в будние дни: 1500 ₽ за день. В выходные: 800 ₽ за день.
- 说明: 错误。图中1500 ₽是成人周末/节假日票价，800 ₽是儿童周末/节假日票价。摘要将两者混淆并错误归类为儿童票价。
- 完整理由: 摘要在前半部分描述准确，但在第4点关于价格的描述中出现了严重的实体错误（将成人周末票价1500元误认为儿童票价，且逻辑混乱）。此外，最后一段出现了严重的重复错误（looping），导致信息完全失真。

**step300-ru-case13** (准确性=2, 类型=entity_error)
- 错句: Справа от видео расположены кнопки лайка (40), комментариев (459), репоста (262) и меню (три точки).
- 说明: 数值与图标对应错误：图中气泡（评论）为40，双箭头（Remix）为459，单箭头（分享）为262，心形（点赞）无数字。摘要完全搞混了这些数值的归属。
- 完整理由: 摘要在描述互动数据时存在明显的实体错误（entity_error）。它将点赞数误报为40（图中点赞图标旁无数字），将评论数误报为459（图中气泡图标旁为40），将转发数误报为262（图中双箭头图标旁为459，单箭头分享图标旁为262）。这种数值与图标的错误挂钩属于严重的虚构与错误归属。

**step300-ru-case16** (准确性=2, 类型=entity_error)
- 错句: Сзади автомобиля стоит женщина с поднятыми руками, выражая радость или удивление.
- 说明: На фото изображен мужчина с длинными волосами, который стоит в открытом люке автомобиля, а не женщина позади него.
- 完整理由: В аннотации допущены серьезные фактические ошибки: неверно определен пол и местоположение человека (мужчина в люке, а не женщина сзади), а также неверно указано количество лайков (1,6 млн вместо 1,3 млн). См. пункты 3 и 7 в проверке фактов.

**step300-ru-case50** (准确性=2, 类型=entity_error)
- 错句: 1. **Действие с тормозом**: Рука человека нажимает на рычаг дискового тормоза, расположенного на вилке велосипеда.
- 说明: На фото рука касается тормозного суппорта (caliper), а не рычага (lever). Кроме того, тормоз находится на заднем перье рамы, а не на вилке.
- 完整理由: В аннотации содержатся существенные фактические ошибки, противоречащие изображению. Во-первых, на тормозном суппорте четко видна надпись «SRAM», а не «SHIMANO». Во-вторых, рука касается самого тормозного механизма (суппорта), а не рычага (ручки) тормоза. В-третьих, тормоз расположен на задней части рамы (рядом с кассетой), а не на вилке велосипеда.

**step300-ru-case73** (准确性=2, 类型=entity_error)
- 错句: Устройство отображает время 21:57, дату 27.07, температуру 26,4°C и влажность 7,27%.
- 说明: 存在两处错误：1. 图中底部的“7.27.”显然是日期（7月27日），摘要将其误认为湿度“7,27%”；2. 图中右上角显示数字“26”，摘要凭空添加了小数位描述为“26,4°C”。
- 完整理由: 摘要在描述数字信息时存在明显的实体错误（entity_error）。它将日期“7.27.”错误地识别为湿度“7,27%”，并且在温度数值中凭空捏造了小数点后的数字“26,4°C”（图中仅可见“26”）。

**step300-ru-case86** (准确性=2, 类型=entity_error)
- 错句: Название видео: «Прожил Пятую Ночь с ПИВОЗАВРОМ в ПОДЪЕЗДЕ! (5 НО...» — указывает на серию контента, связанную с выживанием и юмористическим элементом «пивозавр».
- 说明: 该标题属于下方推荐的视频，主视频标题应为“Прожил Пять Ночей с ПИВОЗАВРОМ в...”。
- 完整理由: 摘要存在多处严重的实质性错误：1. 数量级错误：将 538 个“不喜欢”误报为 538,000（538 тыс.），夸大了三个数量级；2. 实体归属错误：将下方推荐视频的标题误认为主视频标题，并将该视频的播放量（358k）误认为是整个频道的播放量；3. 包含大量主观臆断和元描述（对摘要自身的评价），且结尾句子残缺。

**step300-ru-case96** (准确性=2, 类型=entity_error)
- 错句: На экране видны элементы управления: таймер (12), счётчик очков (50/50), кнопки «10», «10», «10», «50».
- 说明: На кнопках в игре написаны другие значения: 50/50, 20, значок обновления, 10, значок перемотки, 50.
- 完整理由: Хотя общая структура и основные разделы скриншота определены верно, в деталях содержатся грубые фактические ошибки. В разделе игры названия песен и исполнителей сильно искажены (галлюцинации текста): «Как утонули в России» вместо «Как упоительны в России вечера», «Курь Лоза» вместо «Юрий Лоза», «Мк. Седо» вместо «Mr. Credo», «Скины» вместо «Сплин». Также неверно указаны кнопки управления игрой и одна из иконок навигационной панели.

**step300-ru-case97** (准确性=2, 类型=entity_error)
- 错句: Диалог ведётся в мессенджере, где пользователь делится личными воспоминаниями о школьных годах и необычном опыте танцев в лагере, а Антон Максимов отвечает короткими репликами.
- 说明: На скриншоте видна только одна ответная реплика Антона («Пойдем куда нибудь потанцем»), использование множественного числа («репликами») не совсем точно.
- 完整理由: Основная ошибка заключается в неверном определении отправителя ключевого сообщения в конце диалога. Сообщение «Пойдем куда нибудь потанцем» находится в сером облачке слева, что в интерфейсе Telegram однозначно указывает на ответ собеседника (Антона Максимова), в то время как автор резюме приписал его пользователю (отправителю фиолетовых сообщений справа). Это искажает смысл финала переписки. Также неточно указано количество реплик Антона.

**step300-ru-case104** (准确性=2, 类型=grammatical_error)
- 错句: На изображении запечатлены два человека за столом,举杯 в бокалах с красным вином, рядом с ними — блюдо из мяса с луком.
- 说明: 句子中夹杂了中文词汇“举杯”，属于严重的语法/语种错误；同时存在 entity_error，因为右侧的人拿的不是红酒杯。
- 完整理由: 摘要存在明显的语言混合错误（在俄语中夹杂了中文“举杯”），且存在事实性错误：摘要声称两人都拿着装有红酒的高脚杯，但实际上右侧戴面具的人拿着的是一个装有棕色液体的小玻璃杯（烈酒杯），并非红酒高脚杯。

**step300-ru-case110** (准确性=2, 类型=predicate_error)
- 错句: Видео демонстрирует мужчину в кепке и полосатой футболке, указывающего на объект — вероятно, устройство или контейнер с бумагой...
- 说明: 图中男子位于左侧，而指向物体的动作是由右侧画面中的手完成的，这是合拍视频，动作主体归属错误。
- 完整理由: 摘要存在两处明显的硬伤：一是动作归属错误（predicate_error），将右侧画面中手指指向物体的动作归于左侧的男子，实际上这是双人合拍/反应类视频，动作主体并非该男子；二是实体属性错误（entity_error），将纸上明显的韩文（Hangeul）误认为英文。

**step300-ru-case122** (准确性=2, 类型=entity_error)
- 错句: Изображение представляет собой скриншот из социальной сети, где пользователь shahrom_journalist опубликовал сообщение с изображением экрана смартфона, на котором отображается чат с текстовыми сообщениями на русском языке.
- 说明: 聊天消息的语言是塔吉克语，而非俄语（仅系统提示语是俄语）。
- 完整理由: 摘要在描述核心内容（聊天记录）时存在大量严重的事实性错误。它错误地将聊天语言识别为俄语（实际为塔吉克语），错误地转录了几乎所有的时间（如将1:43写成14:49，将08:30写成00:20）、金额（将1500$写成16008）和背景数字（将-436写成-435）。此外，它还将消息内容误认为发送者姓名，并对最后一条关键消息进行了严重的幻觉式转录。

**step300-ru-case126** (准确性=2, 类型=entity_error)
- 错句: Изображение представляет собой страницы из книги А. Корнева «Магия. Сакральные обряды и ритуалы. Практическое руководство»
- 说明: 书名错误，图中显示为“Сильнейшие обряды и ритуалы”，而非“Сакральные”。
- 完整理由: 摘要存在多处严重的事实性错误和幻觉。首先，书名被错误引用（图中为“Сильнейшие”，摘要写成“Сакральные”）。其次，在描述第一个仪式时，将“大米”（рис）误认为“格栅”（решётка），并将“硬币排列成梯形”误描述为“不同面值的纸币”。在第二个仪式的背景故事中，摘要完全颠倒了人物关系：图中是熟人向作者求助，摘要却写成熟人帮助了作者。此外，摘要还虚构了书中不存在的引用语（“祈祷吧，一切都会有的”）和建议（“不要立即检查结果”）。

**step300-ru-case131** (准确性=2, 类型=out_of_context_error)
- 错句: Решение проблем с недвижимостью, земельными спорами, наследственными делами.
- 说明: 这是凭空捏造的信息。原文第一段明确提到的是“逾期或有问题的贷款”（seskhi），而非房地产、土地或遗产。
- 完整理由: 摘要存在严重的幻觉和事实错误。首先，它完全遗漏了信息的核心主题——针对“逾期或有问题的贷款”（格鲁吉亚语：seskhi）以及收债公司的骚扰提供法律援助，反而凭空捏造了“房地产、土地纠纷、遗产”等文中未提及的业务领域。其次，摘要声称文中包含“法院”和“文件”等关键词，但格鲁吉亚语原文中并未出现对应的词汇（sasamarTlo 或 sabutebi）。

**step300-ru-case168** (准确性=2, 类型=entity_error)
- 错句: Автор выражает сильное разочарование и стресс после сдачи зачета по биологии, используя фразы вроде «Истерика въебала» и «Хуй ег о знает».
- 说明: 图中文字为“бх”，通常指生物化学（биохимия），而非生物学（биология）。
- 完整理由: 摘要存在多处实质性事实错误：1. 将“бх”（生物化学）错误识别为“биология”（生物学）；2. 将“общежитие”（宿舍）错误表述为“учебное заведение”（教育机构）；3. 最严重的错误是颠倒了帮助关系，图中文字明确表示作者帮助了朋友们（помогла сдать），而摘要却说作者感谢他们的帮助；4. 摘要中还存在大段内容的重复（见事实核查第11、12条）。

**step300-ru-case177** (准确性=1, 类型=entity_error)
- 错句: Некоторые игроки помечены как травмированные (например, Даниэль Ким, Даниэль Матей), другие — в процессе восстановления (например, Мессиах, Паскаль Гомес).
- 说明: 图中出现的受伤球员是 Гриз Ли 和 Погрыз Дикаприо。摘要中提到的所有姓名均为凭空捏造，图中不存在。
- 完整理由: 摘要包含严重的虚构和技术性错误。首先，摘要中提到的球员姓名（如 Даниэль Ким, Даниэль Матей, Мессиах, Паскаль Гомес）在图片中完全不存在，属于凭空捏造（entity_error）。其次，摘要列出的位置缩写（如 ЛК, ЦК, РК, ЛБ, ЦБ, РБ 等）大部分未在图中出现，图中仅可见 Gk, Ld, Cd, Rd, Lm, Cm, Rm, Cf, Rf。最严重的是，摘要进入了无限重复的循环并被截断，这属于严重的生成故障。

**step300-ru-case183** (准确性=2, 类型=entity_error)
- 错句: Один из участников, одетый в белую майку с надписью «SUNRISE» и номером «1», находится впереди, двое других — позади него, готовые к действию.
- 说明: 图中右侧男子的背心文字是“RUSSIAN BEAR”，而非“SUNRISE”，且无数字“1”；三人基本并排，并非一前两后。
- 完整理由: 摘要在描述主体细节时存在明显的虚构和错误归属。摘要声称其中一名参与者穿着印有“SUNRISE”字样和数字“1”的白色背心，但图片中右侧男子的灰色背心上清晰可见的文字是“RUSSIAN BEAR”，且没有任何数字“1”。这种凭空捏造文字信息的行为属于严重的 entity_error。此外，摘要描述参与者的位置为“一前两后”，而图中三人基本并排站在轮胎后方。

**step300-ru-case197** (准确性=2, 类型=entity_error)
- 错句: Документ является официальным уведомлением от комиссии по делам несовершеннолетних (КДН) о поступлении протокола в отношении гражданина по факту нахождения его ребёнка (РГО) в ночное время без сопровождения.
- 说明: 文中括号内是占位符“(ФИО)”（姓名缩写），而非“РГО”。
- 完整理由: 摘要存在多处严重的事实性错误。首先，将占位符“(ФИО)”（姓名缩写）误读为“РГО”；其次，将地址“ул. Горького 14а каб. 22”错误写成“ул. Горького, д. 22, кв. 22”；再次，将“不出席的后果（в случае неявки）”误解为“再次违规的后果（при повторном нарушении）”；最后，将法律术语“привод”（强制传唤/拘留）误写为“приём”（接待）。此外，摘要还包含大量文中未提及的通用建议。

**step300-ru-case208** (准确性=2, 类型=entity_error)
- 错句: Упоминает ответ Тегерана, заявление СБМГИ и геополитические последствия.
- 说明: 图中文字为“СИБИГИ”（指代政治人物 Сибига），摘要误写为“СБМГИ”。
- 完整理由: 摘要存在多处实质性事实错误：1. 错误归属数值，将第一条视频的分享数“9”误认为第二条视频的评论数；2. 错误扩大范围，声称“两部视频”均标有“18+”和外国代理人提示，但图中仅第一部视频有此提示；3. 文字识别错误，将缩略图中的“СИБИГИ”写成“СБМГИ”；4. 包含大量脱离语境的臆断和建议（如“用户建议”部分），且结尾文字残缺。

**step300-ru-case211** (准确性=2, 类型=entity_error)
- 错句: Левый мужчина носит тактический жилет коричневого цвета поверх рубашки, солнцезащитные очки на голове и имеет короткую стрижку.
- 说明: 左侧男子穿的是一件躯干部分为棕色的作战蛙服（Combat Shirt），而不是在衬衫外面套了战术背心。
- 完整理由: 摘要存在严重的实体错误。首先，右侧男子帽子上的文字清晰可见为“Я НЕ ГРУСТНЫЙ, Я ТРЕЗВЫЙ”（我不难过，我很清醒），而摘要将其错误地识别为“ВООРУЖЕННЫЕ СИЛЫ УКРАИНЫ”（乌克兰武装部队），这属于严重的虚构和错误归属。其次，左侧男子穿着的是躯干为棕色的作战服（Combat Shirt），而非摘要所称的“棕色战术背心”。

</details>

#### zh (中文) — 22 个 badcase

| # | case_id | 错误类型 | 错句（摘要片段） | 评委说明 | 评委理由（截取） |
|---|---------|----------|------------------|----------|-------------------|
| 1 | 17 | entity_error | 这是一张手机聊天记录的截图，对话双方为“bakakay”和“Leilani”，背景为电影《The Batman》。 | 背景图片是游戏/剧集《最后生还者》（The Last of Us）的海报，而非电影《蝙蝠侠》。 | 摘要存在两处明显的硬伤：1. 错误识别了背景图片，图中背景是《最后生还者》（The Last of Us）的乔尔和艾莉，而非《蝙蝠侠》（The Batman）；2. 对关键对话内容的翻译存在严重偏差（幻觉），将“Naa dw ingun roda”（意为：Roda说有）错误解读为“轮子坏了”，将“Puti dw to”（意为：说是白色的）错误解读为因果关系“因为它是白色的”。 |
| 2 | 37 | predicate_error | 用户发送了一条语音消息，时长为9秒，时间为10:21。 | 图中9秒的语音消息位于左侧，是对方发送的，而非用户发送。 | 摘要存在多处实质性事实错误：1. 将对方发送的9秒语音消息错误归属为用户发送（predicate_error）；2. 将10:23正在发送的消息大小“10KB”误认为时长“10秒”（entity_error）；3. 声称所有时间戳均为10:21或10:23，忽略了图中明显的15:22、15:23和10:12（entity_error）。 |
| 3 | 55 | entity_error | 这是一张即时通讯软件（界面特征符合Telegram）的聊天截图，展示了用户与名为“Апаи Саида”（阿帕伊·赛达）的联系人之间的对话记录。 | 界面特征（如右上角的摄像机和电话图标、底部的回形针和相机图标、绿色的发送按钮）明确指向 WhatsApp，而非 Telegram。 | 摘要在关键事实描述上存在多处严重错误：1. 界面识别错误，该界面为 WhatsApp 而非 Telegram；2. 发送方识别完全颠倒，摘要将右侧用户发送的内容误认为对方发送；3. 关键文字理解错误，将系统提示语“3条未读消息”误认为对方回复，将询问“有什么”误认为“你好”；4. 语言识别有误，对话主体为塔吉克语而非俄语。 |
| 4 | 77 | circumstantial_error | 这是一张手机聊天界面截图，显示了一段视频消息。 | 图中显示的是视频播放或社交媒体快拍（Story）查看界面，包含播放进度条和“添加到我的快拍”按钮，而非聊天界面。 | 摘要在关键事实识别上存在多处严重错误。首先，将视频播放/快拍界面误认为“聊天界面”；其次，将图中的塔吉克语（使用西里尔字母）多次误认为俄语；最严重的错误是凭空捏造图中存在“中文”文字，实际上该部分文字依然是塔吉克语。这些错误导致摘要对图片内容的描述严重失真。 |
| 5 | 83 | entity_error | 3. **互动数据**：该帖子显示有138个点赞和4条评论，发布者Amylens TV在6小时前发布。 | 138个点赞和4条评论位于Amylens TV动态的上方，属于前一条动态的数据，而非本条动态的数据。 | 摘要存在明显的实体归属错误。图中显示的“138个点赞和4条评论”位于“Amylens TV”动态的上方，属于上一条动态的互动数据，而非摘要所声称的属于该动态。这种将数据张冠李戴的情况属于实质性事实错误。 |
| 6 | 99 | entity_error | 画面左侧有一名男子身穿白色花纹上衣和黑色裤子，系着红色围巾，正背对镜头向左侧移动。 | 图中男子身上斜跨的是红色绶带，而非围巾。 | 摘要整体描述较为详尽，但在关键数据和对象属性上存在严重错误。首先，点赞数显示为“90,6 тыс.”（俄语缩写，意为9.06万），摘要将其误记为“90.6万”，存在一个数量级的偏差；其次，画面左侧男子身上系的是红色绶带（从肩部斜跨至腰部），摘要将其误认为“围巾”。根据评测准则，数量级错误属于明确的硬伤。 |
| 7 | 102 | predicate_error | 提问内容：下方文字为“Ya-Fattah desangiz nimalar bo'ladi?”，询问“如果每天喝313杯水会发生什么？”。 | 严重错误。“Ya-Fattah desangiz”意为“如果你说/念诵‘Ya-Fattah’”，而非“喝水”。原意是对着1杯水念诵313次名号，摘要将其误解为“喝313杯水”，行为主体动作和数量逻辑均错误。 | 摘要在解读核心文字内容时存在严重错误。图中文字“Ya-Fattah desangiz”意为“如果你念诵‘Ya-Fattah’（真主名号）”，摘要将其错误地解读为“喝313杯水”，这属于严重的谓词错误和事实扭曲，完全改变了视频传达的宗教/民俗语境。 |
| 8 | 128 | entity_error | 视频画面：背景为一名身穿西装的男子（马克·泽尔曼）在听证会或正式场合发言，神情严肃。 | 图中人物是迈克尔·科恩（Michael Cohen），而非“马克·泽尔曼”，存在明显的实体识别错误。 | 摘要在描述视频画面时出现了严重的实体错误，将图中极其知名的公众人物迈克尔·科恩（Michael Cohen）错误识别为“马克·泽尔曼”（Mark Zelman），这属于典型的张冠李戴。除此之外，摘要对文字内容的翻译和场景的解读基本准确。 |
| 9 | 136 | entity_error | 收到一笔来自MPESA的转账，参考号为UGF3FBWAFR，金额为453855.00 KES，交易时间为2026年7月15日23:30:55。 | 图片中 7月15日的短信并未显示金额。453855.00 KES 实际上是 7月24日短信中显示的账户余额，摘要将其错误归属为交易金额。 | 摘要存在严重的实体错误（entity_error）。它将 2026年7月24日查询到的账户余额（453855.00 KES）错误地归属为 2026年7月15日 MPESA 转账的交易金额。图片中 7月15日的短信仅显示了参考号和时间，并未显示具体金额。这种张冠李戴属于实质性的事实错误。 |
| 10 | 144 | circumstantial_error | 这是一张社交媒体（疑似TikTok）的评论区截图，内容主要围绕手机话费充值、网络信号及工作祝福展开。 | 界面 UI（如心形点赞图标、回复按钮样式、翻译链接）明显属于 Instagram 而非 TikTok。此外，顶部文字的核心是“设备欠款（deni la kifaa）”而非单纯的“话费充值”。 | 摘要在翻译和识别关键信息方面存在多处严重错误。首先，将界面误认为 TikTok（实际为 Instagram）；其次，在顶部文字中凭空捏造了图中未出现的“M-Pesa”；最严重的是对斯瓦希里语评论的翻译完全错误：将“祝你受祝福”翻译为“生日快乐”，将“孩子的医生”翻译为“丈夫的医生”，且将“寻找孩子的父亲”误读为“正在等待”。此外，将“设备欠款（deni la kifaa）”概括为“话费充值”也不够准确。 |
| 11 | 145 | entity_error | 这是一张名为Ian Joshua Cariño的即时通讯软件聊天截图，对话内容涉及工作交接、薪资结算及车辆租赁事宜。 | 对话内容并非“薪资结算”和“车辆租赁”，而是关于借贷利息（tubo）和摩托车（Click motor）出借的讨论。 | 摘要存在多处实质性事实错误。首先，将“tubo”（利息/利润）误认为“薪资”和“提成”，文中讨论的是借贷利息而非工资结算。其次，将“papahiram”（借出）误认为“租赁”，且将关于支付日期的讨论误认为“确认收到款项”。最严重的错误是将摩托车品牌型号“Click”（本田的一款摩托车）字面翻译为“可点击的”，属于严重的实体识别错误。 |
| 12 | 155 | entity_error | 用户Okafor Chiemeka发送了“2/9 May 2026 Football Attendance”（2026年5月2日足球考勤）信息，其中包含“Abel Chair”的考勤状态（显示为对勾、叉号及大拇指表情）。 | 图中显示的表情符号是✅（对勾）和❌（叉号），并没有大拇指表情。 | 摘要存在多处严重的事实性错误：1. 错误归属了语音消息和文字消息“My man my man”的发送者（图中语音是Emeka发的，文字是Okafor发的）；2. 将图片文件大小（32 kB）误认为交易金额；3. 错误描述了考勤状态中的表情符号（图中是✅和❌，没有大拇指）。 |
| 13 | 159 | predicate_error | 2. 进球信息：巴里什在第45+6分钟（乌龙球）为斯帕塔克特纳瓦得分，普多霍罗茨基在第41分钟为斯卡利卡扳平比分。 | 逻辑错误。普多霍罗茨基在41分钟进球，此时比分应为0-1，他是首开纪录者；巴里什在45+6分钟的乌龙球才使比分变为1-1（扳平）。 | 摘要存在两处明显的逻辑与事实错误：第一，进球顺序描述错误，普多霍罗茨基在41分钟进球，早于45+6分钟的乌龙球，因此他进的是领先球而非扳平球；第二，机会统计描述自相矛盾且与图不符，图中显示斯卡利卡有1次大机会，摘要却称“双方均无大机会”。 |
| 14 | 160 | entity_error | 人物：图片中的主角为足球运动员凯文·德布劳内（Kevin De Bruyne）。 | 图中球员是莱尼·约罗（Leny Yoro），并非凯文·德布劳内，两者外貌特征完全不同。 | 摘要存在严重的实体错误（entity_error），将图片中的黑人球员错误识别为白人球员凯文·德布劳内（Kevin De Bruyne）。实际上，图中球员是莱尼·约罗（Leny Yoro）。这种身份识别的张冠李戴属于实质性事实错误。 |
| 15 | 161 | entity_error | 这是一张社交媒体视频截图，内容围绕足球运动员奥雷利安·楚阿梅尼（Aurélien Tchouaméni）展开。 | 图中球员是奥斯曼·登贝莱（Ousmane Dembélé），而非楚阿梅尼。 | 摘要在核心人物识别上存在严重错误。图中展示的足球运动员是奥斯曼·登贝莱（Ousmane Dembélé），而非摘要中所称的奥雷利安·楚阿梅尼（Aurélien Tchouaméni）。图中球员身穿的法国国家队7号球衣以及巴黎圣日耳曼（PSG）球衣均是登贝莱的特征，楚阿梅尼在国家队身穿8号且效力于皇家马德里。这一实体错误贯穿全文。 |
| 16 | 166 | entity_error | 这是一张关于台湾政治人物戴瑗姍的短视频截图，内容涉及对她在日菲海域谈判争议事件中的表现进行批评。 | 图中文字明确显示人物姓名为“戴瑋姍”，而非“戴瑗姍”。 | 摘要存在多处实质性错误：首先，核心人物姓名在图中清晰显示为“戴瑋姍”，摘要却多次错误写为“戴瑗姍”；其次，摘要称“炸锅”、“不认输”等词在评论区可见，实际上这些词是视频画面中的蓝色横幅字幕内容，图中并未显示评论区具体内容；最后，摘要的第7、8点及部分结论属于对视频影响和建议的凭空臆断，图中无任何依据。 |
| 17 | 168 | entity_error | 发布者信息：帖子由名为“አዲስ ሰሜን”（Adis Semien）的账号发布，发布时间为22小时前。 | 账号名称错误。图中显示为“አዲስ መረጃ”（Addis Mereja），而非“አዲስ ሰሜን”。 | 摘要在处理图内文字信息时出现了严重的实体错误（entity_error）。首先，发布者账号名称被错误识别为“አዲስ ሰሜን”（Adis Semien），而图中清晰显示为“አዲስ መረጃ”（Addis Mereja）；其次，标签被错误识别为“#ዘፈን_ዘፈን”（歌曲/音乐），而图中实际为“#ዜና_ሹመት”（意为“任命新闻”）。这些错误导致对帖子主题的理解完全偏差。 |
| 18 | 170 | entity_error | 报道聚焦于菲律宾前总统费迪南德·马科斯（Ferdinand Marcos Jr.）的发言人（BBM）针对防洪项目腐败指控的回应。 | BBM 指的是现任总统小马科斯，而非前总统的发言人。 | 摘要在核心事实理解上存在多处严重错误：1. 身份误认：BBM（Bongbong Marcos）是现任菲律宾总统，而非前总统的发言人；2. 关键术语错误：FPRRD 指代前总统杜特尔特（Former President Rodrigo Roa Duterte），而非所谓的“复兴党”；3. 核心语义颠倒：图中文字意为“BBM称杜特尔特在打击防洪项目腐败方面毫无作为”，摘要却解读为“没有腐败行为”，完全扭曲了新闻原意；4. 数据错误：点赞数为332K，摘要写成33.2K，相差一个数量级。 |
| 19 | 173 | predicate_error | 核心观点：库里亚呼吁执政党（UDA）应回归其竞选承诺，即改善民生、建设道路并推动国家发展。 | 图中文字是库里亚在DCP获胜后嘲讽选民，让他们退还UDA的福利（煤气罐），并质疑DCP议员是否能兑现承诺，而非呼吁UDA回归承诺。 | 摘要严重误读了图片中文字的政治含义和人物立场。图中文字显示莫西斯·库里亚（Moses Kuria）是在嘲讽选民选择了DCP的议员，并要求他们退还UDA提供的煤气罐，质疑新议员是否能带来发展。摘要却将其解读为库里亚呼吁UDA回归承诺或支持DCP，这与图片传达的对抗性语境完全相反。 |
| 20 | 176 | entity_error | 埃塞俄比亚广播公司发布关于2025年4月25日（埃塞俄比亚历法）的官方声明，并配发相关会议现场图片。 | 图中文字“25.4 ቢሊዮን ብር”意为“25.4亿比尔”，是金额而非日期。摘要将其误认为日期属于严重的实体错误。 | 摘要存在多处严重的实体错误和语境脱离错误。首先，摘要将图中的金额“25.4 亿比尔”错误解读为日期“2025年4月25日”，这是严重的数字识别错误；其次，摘要将名牌上的文字错误翻译为“和平、和平、和平”，实际上那是人名；此外，摘要声称图片是“官方声明”，但图中文字明确显示是关于预算批准的新闻报道。基于这些实质性错误，准确性评为2分。 |
| 21 | 214 | entity_error | 在她身后，一只巨大的恐龙正低头注视着她。 | 图中老妇人身后是一个巨大的人手（可见手指和指甲），而非恐龙。 | 摘要在描述画面核心元素时出现了严重的实体错误。它将画面中非常明显的“巨大的人手”误认为是“一只巨大的恐龙”，这属于与图片事实明确矛盾的硬伤。其他关于界面类型、老妇人动作及文案的描述是正确的。 |
| 22 | 218 | entity_error | 区域差异：高温主要集中在东部地区（如杰贝尔·阿赫马尔），而西部和南部地区气温相对较低。 | 图中文字提到的高温地区是阿齐齐亚（Al-Aziziya）和贾法拉（Jafara），这两个地区位于利比亚西北部（西部），而非东部。图中也未提及杰贝尔·阿赫马尔。 | 摘要存在多处严重的实体错误和事实性错误。首先，摘要将高温区域错误地归为东部，而图中文字明确提到的 العزيزية（阿齐齐亚）和 جفارة（贾法拉）均位于利比亚西部；其次，摘要将地图上的气温数据错误归属给的黎波里和班加西，实际上地图显示的是意大利的卡利亚里（28°C）和希腊（30°C）。 |

<details><summary>展开各 badcase 完整理由</summary>

**step300-zh-case17** (准确性=2, 类型=entity_error)
- 错句: 这是一张手机聊天记录的截图，对话双方为“bakakay”和“Leilani”，背景为电影《The Batman》。
- 说明: 背景图片是游戏/剧集《最后生还者》（The Last of Us）的海报，而非电影《蝙蝠侠》。
- 完整理由: 摘要存在两处明显的硬伤：1. 错误识别了背景图片，图中背景是《最后生还者》（The Last of Us）的乔尔和艾莉，而非《蝙蝠侠》（The Batman）；2. 对关键对话内容的翻译存在严重偏差（幻觉），将“Naa dw ingun roda”（意为：Roda说有）错误解读为“轮子坏了”，将“Puti dw to”（意为：说是白色的）错误解读为因果关系“因为它是白色的”。

**step300-zh-case37** (准确性=2, 类型=predicate_error)
- 错句: 用户发送了一条语音消息，时长为9秒，时间为10:21。
- 说明: 图中9秒的语音消息位于左侧，是对方发送的，而非用户发送。
- 完整理由: 摘要存在多处实质性事实错误：1. 将对方发送的9秒语音消息错误归属为用户发送（predicate_error）；2. 将10:23正在发送的消息大小“10KB”误认为时长“10秒”（entity_error）；3. 声称所有时间戳均为10:21或10:23，忽略了图中明显的15:22、15:23和10:12（entity_error）。

**step300-zh-case55** (准确性=2, 类型=entity_error)
- 错句: 这是一张即时通讯软件（界面特征符合Telegram）的聊天截图，展示了用户与名为“Апаи Саида”（阿帕伊·赛达）的联系人之间的对话记录。
- 说明: 界面特征（如右上角的摄像机和电话图标、底部的回形针和相机图标、绿色的发送按钮）明确指向 WhatsApp，而非 Telegram。
- 完整理由: 摘要在关键事实描述上存在多处严重错误：1. 界面识别错误，该界面为 WhatsApp 而非 Telegram；2. 发送方识别完全颠倒，摘要将右侧用户发送的内容误认为对方发送；3. 关键文字理解错误，将系统提示语“3条未读消息”误认为对方回复，将询问“有什么”误认为“你好”；4. 语言识别有误，对话主体为塔吉克语而非俄语。

**step300-zh-case77** (准确性=2, 类型=circumstantial_error)
- 错句: 这是一张手机聊天界面截图，显示了一段视频消息。
- 说明: 图中显示的是视频播放或社交媒体快拍（Story）查看界面，包含播放进度条和“添加到我的快拍”按钮，而非聊天界面。
- 完整理由: 摘要在关键事实识别上存在多处严重错误。首先，将视频播放/快拍界面误认为“聊天界面”；其次，将图中的塔吉克语（使用西里尔字母）多次误认为俄语；最严重的错误是凭空捏造图中存在“中文”文字，实际上该部分文字依然是塔吉克语。这些错误导致摘要对图片内容的描述严重失真。

**step300-zh-case83** (准确性=2, 类型=entity_error)
- 错句: 3. **互动数据**：该帖子显示有138个点赞和4条评论，发布者Amylens TV在6小时前发布。
- 说明: 138个点赞和4条评论位于Amylens TV动态的上方，属于前一条动态的数据，而非本条动态的数据。
- 完整理由: 摘要存在明显的实体归属错误。图中显示的“138个点赞和4条评论”位于“Amylens TV”动态的上方，属于上一条动态的互动数据，而非摘要所声称的属于该动态。这种将数据张冠李戴的情况属于实质性事实错误。

**step300-zh-case99** (准确性=2, 类型=entity_error)
- 错句: 画面左侧有一名男子身穿白色花纹上衣和黑色裤子，系着红色围巾，正背对镜头向左侧移动。
- 说明: 图中男子身上斜跨的是红色绶带，而非围巾。
- 完整理由: 摘要整体描述较为详尽，但在关键数据和对象属性上存在严重错误。首先，点赞数显示为“90,6 тыс.”（俄语缩写，意为9.06万），摘要将其误记为“90.6万”，存在一个数量级的偏差；其次，画面左侧男子身上系的是红色绶带（从肩部斜跨至腰部），摘要将其误认为“围巾”。根据评测准则，数量级错误属于明确的硬伤。

**step300-zh-case102** (准确性=2, 类型=predicate_error)
- 错句: 提问内容：下方文字为“Ya-Fattah desangiz nimalar bo'ladi?”，询问“如果每天喝313杯水会发生什么？”。
- 说明: 严重错误。“Ya-Fattah desangiz”意为“如果你说/念诵‘Ya-Fattah’”，而非“喝水”。原意是对着1杯水念诵313次名号，摘要将其误解为“喝313杯水”，行为主体动作和数量逻辑均错误。
- 完整理由: 摘要在解读核心文字内容时存在严重错误。图中文字“Ya-Fattah desangiz”意为“如果你念诵‘Ya-Fattah’（真主名号）”，摘要将其错误地解读为“喝313杯水”，这属于严重的谓词错误和事实扭曲，完全改变了视频传达的宗教/民俗语境。

**step300-zh-case128** (准确性=2, 类型=entity_error)
- 错句: 视频画面：背景为一名身穿西装的男子（马克·泽尔曼）在听证会或正式场合发言，神情严肃。
- 说明: 图中人物是迈克尔·科恩（Michael Cohen），而非“马克·泽尔曼”，存在明显的实体识别错误。
- 完整理由: 摘要在描述视频画面时出现了严重的实体错误，将图中极其知名的公众人物迈克尔·科恩（Michael Cohen）错误识别为“马克·泽尔曼”（Mark Zelman），这属于典型的张冠李戴。除此之外，摘要对文字内容的翻译和场景的解读基本准确。

**step300-zh-case136** (准确性=2, 类型=entity_error)
- 错句: 收到一笔来自MPESA的转账，参考号为UGF3FBWAFR，金额为453855.00 KES，交易时间为2026年7月15日23:30:55。
- 说明: 图片中 7月15日的短信并未显示金额。453855.00 KES 实际上是 7月24日短信中显示的账户余额，摘要将其错误归属为交易金额。
- 完整理由: 摘要存在严重的实体错误（entity_error）。它将 2026年7月24日查询到的账户余额（453855.00 KES）错误地归属为 2026年7月15日 MPESA 转账的交易金额。图片中 7月15日的短信仅显示了参考号和时间，并未显示具体金额。这种张冠李戴属于实质性的事实错误。

**step300-zh-case144** (准确性=2, 类型=circumstantial_error)
- 错句: 这是一张社交媒体（疑似TikTok）的评论区截图，内容主要围绕手机话费充值、网络信号及工作祝福展开。
- 说明: 界面 UI（如心形点赞图标、回复按钮样式、翻译链接）明显属于 Instagram 而非 TikTok。此外，顶部文字的核心是“设备欠款（deni la kifaa）”而非单纯的“话费充值”。
- 完整理由: 摘要在翻译和识别关键信息方面存在多处严重错误。首先，将界面误认为 TikTok（实际为 Instagram）；其次，在顶部文字中凭空捏造了图中未出现的“M-Pesa”；最严重的是对斯瓦希里语评论的翻译完全错误：将“祝你受祝福”翻译为“生日快乐”，将“孩子的医生”翻译为“丈夫的医生”，且将“寻找孩子的父亲”误读为“正在等待”。此外，将“设备欠款（deni la kifaa）”概括为“话费充值”也不够准确。

**step300-zh-case145** (准确性=2, 类型=entity_error)
- 错句: 这是一张名为Ian Joshua Cariño的即时通讯软件聊天截图，对话内容涉及工作交接、薪资结算及车辆租赁事宜。
- 说明: 对话内容并非“薪资结算”和“车辆租赁”，而是关于借贷利息（tubo）和摩托车（Click motor）出借的讨论。
- 完整理由: 摘要存在多处实质性事实错误。首先，将“tubo”（利息/利润）误认为“薪资”和“提成”，文中讨论的是借贷利息而非工资结算。其次，将“papahiram”（借出）误认为“租赁”，且将关于支付日期的讨论误认为“确认收到款项”。最严重的错误是将摩托车品牌型号“Click”（本田的一款摩托车）字面翻译为“可点击的”，属于严重的实体识别错误。

**step300-zh-case155** (准确性=2, 类型=entity_error)
- 错句: 用户Okafor Chiemeka发送了“2/9 May 2026 Football Attendance”（2026年5月2日足球考勤）信息，其中包含“Abel Chair”的考勤状态（显示为对勾、叉号及大拇指表情）。
- 说明: 图中显示的表情符号是✅（对勾）和❌（叉号），并没有大拇指表情。
- 完整理由: 摘要存在多处严重的事实性错误：1. 错误归属了语音消息和文字消息“My man my man”的发送者（图中语音是Emeka发的，文字是Okafor发的）；2. 将图片文件大小（32 kB）误认为交易金额；3. 错误描述了考勤状态中的表情符号（图中是✅和❌，没有大拇指）。

**step300-zh-case159** (准确性=2, 类型=predicate_error)
- 错句: 2. 进球信息：巴里什在第45+6分钟（乌龙球）为斯帕塔克特纳瓦得分，普多霍罗茨基在第41分钟为斯卡利卡扳平比分。
- 说明: 逻辑错误。普多霍罗茨基在41分钟进球，此时比分应为0-1，他是首开纪录者；巴里什在45+6分钟的乌龙球才使比分变为1-1（扳平）。
- 完整理由: 摘要存在两处明显的逻辑与事实错误：第一，进球顺序描述错误，普多霍罗茨基在41分钟进球，早于45+6分钟的乌龙球，因此他进的是领先球而非扳平球；第二，机会统计描述自相矛盾且与图不符，图中显示斯卡利卡有1次大机会，摘要却称“双方均无大机会”。

**step300-zh-case160** (准确性=2, 类型=entity_error)
- 错句: 人物：图片中的主角为足球运动员凯文·德布劳内（Kevin De Bruyne）。
- 说明: 图中球员是莱尼·约罗（Leny Yoro），并非凯文·德布劳内，两者外貌特征完全不同。
- 完整理由: 摘要存在严重的实体错误（entity_error），将图片中的黑人球员错误识别为白人球员凯文·德布劳内（Kevin De Bruyne）。实际上，图中球员是莱尼·约罗（Leny Yoro）。这种身份识别的张冠李戴属于实质性事实错误。

**step300-zh-case161** (准确性=2, 类型=entity_error)
- 错句: 这是一张社交媒体视频截图，内容围绕足球运动员奥雷利安·楚阿梅尼（Aurélien Tchouaméni）展开。
- 说明: 图中球员是奥斯曼·登贝莱（Ousmane Dembélé），而非楚阿梅尼。
- 完整理由: 摘要在核心人物识别上存在严重错误。图中展示的足球运动员是奥斯曼·登贝莱（Ousmane Dembélé），而非摘要中所称的奥雷利安·楚阿梅尼（Aurélien Tchouaméni）。图中球员身穿的法国国家队7号球衣以及巴黎圣日耳曼（PSG）球衣均是登贝莱的特征，楚阿梅尼在国家队身穿8号且效力于皇家马德里。这一实体错误贯穿全文。

**step300-zh-case166** (准确性=2, 类型=entity_error)
- 错句: 这是一张关于台湾政治人物戴瑗姍的短视频截图，内容涉及对她在日菲海域谈判争议事件中的表现进行批评。
- 说明: 图中文字明确显示人物姓名为“戴瑋姍”，而非“戴瑗姍”。
- 完整理由: 摘要存在多处实质性错误：首先，核心人物姓名在图中清晰显示为“戴瑋姍”，摘要却多次错误写为“戴瑗姍”；其次，摘要称“炸锅”、“不认输”等词在评论区可见，实际上这些词是视频画面中的蓝色横幅字幕内容，图中并未显示评论区具体内容；最后，摘要的第7、8点及部分结论属于对视频影响和建议的凭空臆断，图中无任何依据。

**step300-zh-case168** (准确性=2, 类型=entity_error)
- 错句: 发布者信息：帖子由名为“አዲስ ሰሜን”（Adis Semien）的账号发布，发布时间为22小时前。
- 说明: 账号名称错误。图中显示为“አዲስ መረጃ”（Addis Mereja），而非“አዲስ ሰሜን”。
- 完整理由: 摘要在处理图内文字信息时出现了严重的实体错误（entity_error）。首先，发布者账号名称被错误识别为“አዲስ ሰሜን”（Adis Semien），而图中清晰显示为“አዲስ መረጃ”（Addis Mereja）；其次，标签被错误识别为“#ዘፈን_ዘፈን”（歌曲/音乐），而图中实际为“#ዜና_ሹመት”（意为“任命新闻”）。这些错误导致对帖子主题的理解完全偏差。

**step300-zh-case170** (准确性=2, 类型=entity_error)
- 错句: 报道聚焦于菲律宾前总统费迪南德·马科斯（Ferdinand Marcos Jr.）的发言人（BBM）针对防洪项目腐败指控的回应。
- 说明: BBM 指的是现任总统小马科斯，而非前总统的发言人。
- 完整理由: 摘要在核心事实理解上存在多处严重错误：1. 身份误认：BBM（Bongbong Marcos）是现任菲律宾总统，而非前总统的发言人；2. 关键术语错误：FPRRD 指代前总统杜特尔特（Former President Rodrigo Roa Duterte），而非所谓的“复兴党”；3. 核心语义颠倒：图中文字意为“BBM称杜特尔特在打击防洪项目腐败方面毫无作为”，摘要却解读为“没有腐败行为”，完全扭曲了新闻原意；4. 数据错误：点赞数为332K，摘要写成33.2K，相差一个数量级。

**step300-zh-case173** (准确性=2, 类型=predicate_error)
- 错句: 核心观点：库里亚呼吁执政党（UDA）应回归其竞选承诺，即改善民生、建设道路并推动国家发展。
- 说明: 图中文字是库里亚在DCP获胜后嘲讽选民，让他们退还UDA的福利（煤气罐），并质疑DCP议员是否能兑现承诺，而非呼吁UDA回归承诺。
- 完整理由: 摘要严重误读了图片中文字的政治含义和人物立场。图中文字显示莫西斯·库里亚（Moses Kuria）是在嘲讽选民选择了DCP的议员，并要求他们退还UDA提供的煤气罐，质疑新议员是否能带来发展。摘要却将其解读为库里亚呼吁UDA回归承诺或支持DCP，这与图片传达的对抗性语境完全相反。

**step300-zh-case176** (准确性=2, 类型=entity_error)
- 错句: 埃塞俄比亚广播公司发布关于2025年4月25日（埃塞俄比亚历法）的官方声明，并配发相关会议现场图片。
- 说明: 图中文字“25.4 ቢሊዮን ብር”意为“25.4亿比尔”，是金额而非日期。摘要将其误认为日期属于严重的实体错误。
- 完整理由: 摘要存在多处严重的实体错误和语境脱离错误。首先，摘要将图中的金额“25.4 亿比尔”错误解读为日期“2025年4月25日”，这是严重的数字识别错误；其次，摘要将名牌上的文字错误翻译为“和平、和平、和平”，实际上那是人名；此外，摘要声称图片是“官方声明”，但图中文字明确显示是关于预算批准的新闻报道。基于这些实质性错误，准确性评为2分。

**step300-zh-case214** (准确性=2, 类型=entity_error)
- 错句: 在她身后，一只巨大的恐龙正低头注视着她。
- 说明: 图中老妇人身后是一个巨大的人手（可见手指和指甲），而非恐龙。
- 完整理由: 摘要在描述画面核心元素时出现了严重的实体错误。它将画面中非常明显的“巨大的人手”误认为是“一只巨大的恐龙”，这属于与图片事实明确矛盾的硬伤。其他关于界面类型、老妇人动作及文案的描述是正确的。

**step300-zh-case218** (准确性=2, 类型=entity_error)
- 错句: 区域差异：高温主要集中在东部地区（如杰贝尔·阿赫马尔），而西部和南部地区气温相对较低。
- 说明: 图中文字提到的高温地区是阿齐齐亚（Al-Aziziya）和贾法拉（Jafara），这两个地区位于利比亚西北部（西部），而非东部。图中也未提及杰贝尔·阿赫马尔。
- 完整理由: 摘要存在多处严重的实体错误和事实性错误。首先，摘要将高温区域错误地归为东部，而图中文字明确提到的 العزيزية（阿齐齐亚）和 جفارة（贾法拉）均位于利比亚西部；其次，摘要将地图上的气温数据错误归属给的黎波里和班加西，实际上地图显示的是意大利的卡利亚里（28°C）和希腊（30°C）。

</details>

---

## 6. 典型错误模式与改进建议

### 6.1 数值/数量级识别错误（entity_error，最频繁）

- **症状**：把 `105.3K`（10.53 万）写成 `105.3万`（放大 10×）；`79K`（7.9 万）写成 `7.9千`（缩小 10×）；`222K` 写成 `22K`；电表 `6.40 kWh` 写成 `640 kWh`；`5(80)A` 写成 `6/80A`；`109` 关卡号误为分数。
- **根因**：模型对 K/M 单位换算不严格，对小数点位置敏感度低；UI 数字的图标归属推断不够严谨。
- **改进**：在 summary 训练数据中增加互动数据 K/M 换算的负例；在 prompt 中要求逐个列出「图标→指标→数值」的对应关系后再整合。

### 6.2 消息发送方归属颠倒（predicate_error）

- **症状**：WhatsApp/Telegram 界面中左右气泡方向判错，把对方发的语音/文字说成用户发的，把 9 秒语音归属到错误的人；`00:28`（凌晨 12:28 发送时间）误为视频时长。
- **根因**：模型未显式利用「左气泡=对方，右气泡=自己」这一稳定 UI 约定。
- **改进**：训练数据补 IM 界面的发送方显式标注；prompt 中要求先标「左/右气泡→发送方」再总结。

### 6.3 平台/界面识别错误（circumstantial_error）

- **症状**：把 Instagram Reels 识别为 TikTok；把 WhatsApp 识别为 Telegram；把短信/SMS 界面识别为 WhatsApp；把 TikTok 观看历史识别为 Instagram。
- **根因**：模型对不同平台 UI 细节差异（底部导航栏、绿色顶栏、相机/电话图标位置）不敏感。
- **改进**：训练集中补多平台 UI 标注样本；prompt 中要求先识别「界面类型+依据」。

### 6.4 语种识别错误（entity_error）

- **症状**：把库尔德语（西里尔字母）误为波斯语；塔吉克语（西里尔字母）误为俄语；印尼语俚语 `Boles`（=Boleh，意为"可以"）误为"身体不适"；`jam set 8`（印尼语"7:30"）误为"8 PM"。
- **根因**：模型对小语种（库尔德语、塔吉克语）和俚语缩写缺乏知识；倾向于套用主流语种（波斯语、俄语）的解释。
- **改进**：训练数据补充小语种 + 俚语样本；prompt 中要求先做语种识别并给出依据。

### 6.5 幻觉/捏造（out_of_context_error）

- **症状**：把视频标题误为另一视频标题；把背景文字 `LA FORGE DES CHANSONS` 误为 `L'ORGANISATION DES CHANSONS`；把电表型号 `DDSD101` 抄成 `DGBD101`；凭空捏造文档标题 `FINANCEMENT PAR L'ÉTAT`。
- **根因**：模型在转录可见文字时"脑补"近似词而非严格逐字转录；在缺乏明确文字时倾向于合理化猜测。
- **改进**：训练数据补 OCR 严格转录任务；prompt 中要求"逐字转录图内可见外文原文"后再翻译。

### 6.6 重复/looping 生成

- **症状**：摘要末尾出现重复短语或循环内容，多见于 ru 语种 step300 case 7。
- **根因**：解码 repetition penalty 不够，或模型在 summary 长度约束下的退化。
- **改进**：vllm serve 增加 `--repetition-penalty` 或调整 `frequency_penalty`；训练数据剔除重复段落。

---

## 7. step150 vs step300 对比结论

| 维度 | step150 | step300 | 差值 |
|------|---------|---------|------|
| 准确性 | 4.265 | 4.306 | +0.041 |
| 简洁性 | 4.555 | 4.571 | +0.017 |
| 完整性 | 4.934 | 4.937 | +0.003 |
| 格式 | 4.961 | 4.967 | +0.006 |
| 语种遵循度 | 1.000 | 0.999 | -0.001 |
| **总均分** | **4.679** | **4.695** | **+0.016** |

**关键观察**：
1. step300 相比 step150 在所有维度上**几乎无差异**（差值均在 ±0.05 量级），准确性 +0.041、简洁 +0.016，但 zh 准确性反而下降 0.081。
2. badcase 数量在两个 ckpt 完全相同（76 vs 76），错误模式也一致，说明训练并未针对性修复这些 case。
3. 推测原因：RP-OPSD v3_no_ema 的 EMA 关闭后，自蒸馏信号变弱；从 step150 到 step300 的 150 步训练可能只是平滑收敛，未引入新的能力增量。
4. 建议：是否继续训练到 step450/600 看是否出现拐点；或对比 v3（带 EMA）版本看 EMA 是否对 badcase 修复有实质帮助。

---

## 8. 评测产物

- 推理结果：`/data4/wumeimei/flash_note/infer/infer_res_0904/flashnote_{lang}_rp_opsd_v3noema_summary_9b_step{150,300}.json`
- MOS 评测 JSON：`/data4/wumeimei/flash_note/eval_results/eval_res_0904/rp_opsd_v3noema_summary_9b_step{150,300}/{en,fr,ru,zh}/summary_mos_results.json`
- 评测 log：`/data4/wumeimei/flash_note/infer/logs/mos_v3_m2_step{150,300}_0904_0920.log`
- Merged ckpt：`/data4/wumeimei/flash_note/RP-OPSD/outputs/flashnote_train_v3_no_ema/merged/step_{150,300}_m2/`

报告生成时间: 2026-09-04 12:00
