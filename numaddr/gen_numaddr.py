#!/usr/bin/env python3
"""Independent Chinese eval set: number-only addresses vs. non-address numbers.

Usage: python -I gen_numaddr.py OUT.jsonl
Deterministic for a given seed (20261016). Pure standard library; all word
lists and templates are hand-written in this file.

Template syntax: Python str.format placeholders for fillers; the gold address
span is wrapped in [[ ... ]] (markers are removed and offsets recorded after
formatting). Fillers never contain "[[" or "]]".
"""
import collections
import hashlib
import json
import random
import re
import sys

SEED = 20261016
N_POS, N_NEG, N_MIX = 550, 550, 100

# ---------------------------------------------------------------- word lists
SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤")
FRUITS = ["荔枝", "杨梅", "樱桃", "水蜜桃", "葡萄", "芒果", "西瓜", "橘子"]
COLORS = ["红", "绿", "蓝", "黑", "白", "灰"]

# (province, city, districts); municipalities use an empty province.
PROV_CITY = [
    ("", "北京市", "朝阳区 海淀区 西城区 东城区 丰台区 通州区 昌平区"),
    ("", "上海市", "浦东新区 徐汇区 静安区 黄浦区 长宁区 闵行区 杨浦区"),
    ("", "天津市", "和平区 河西区 南开区 河东区 滨海新区"),
    ("", "重庆市", "渝中区 江北区 南岸区 沙坪坝区 九龙坡区 渝北区"),
    ("广东省", "广州市", "天河区 越秀区 海珠区 白云区 番禺区 荔湾区"),
    ("广东省", "深圳市", "南山区 福田区 罗湖区 宝安区 龙岗区 龙华区"),
    ("广东省", "佛山市", "禅城区 南海区 顺德区"),
    ("浙江省", "杭州市", "西湖区 拱墅区 上城区 滨江区 余杭区 萧山区"),
    ("浙江省", "宁波市", "海曙区 江北区 鄞州区 北仑区"),
    ("浙江省", "温州市", "鹿城区 龙湾区 瓯海区"),
    ("江苏省", "南京市", "鼓楼区 玄武区 秦淮区 建邺区 江宁区 栖霞区"),
    ("江苏省", "苏州市", "姑苏区 吴中区 相城区 虎丘区"),
    ("江苏省", "无锡市", "梁溪区 滨湖区 新吴区 锡山区"),
    ("山东省", "济南市", "历下区 市中区 槐荫区 天桥区 历城区"),
    ("山东省", "青岛市", "市南区 市北区 李沧区 崂山区 黄岛区"),
    ("四川省", "成都市", "锦江区 青羊区 金牛区 武侯区 成华区"),
    ("湖北省", "武汉市", "江汉区 江岸区 武昌区 洪山区 汉阳区"),
    ("湖南省", "长沙市", "岳麓区 芙蓉区 天心区 开福区 雨花区"),
    ("河南省", "郑州市", "金水区 中原区 二七区 管城回族区 惠济区"),
    ("河北省", "石家庄市", "长安区 桥西区 新华区 裕华区"),
    ("陕西省", "西安市", "雁塔区 碑林区 莲湖区 新城区 未央区"),
    ("福建省", "厦门市", "思明区 湖里区 集美区 海沧区"),
    ("福建省", "福州市", "鼓楼区 台江区 仓山区 晋安区"),
    ("安徽省", "合肥市", "蜀山区 包河区 庐阳区 瑶海区"),
    ("江西省", "南昌市", "东湖区 西湖区 青山湖区 红谷滩区"),
    ("辽宁省", "沈阳市", "和平区 沈河区 皇姑区 铁西区 浑南区"),
    ("辽宁省", "大连市", "中山区 西岗区 沙河口区 甘井子区"),
    ("吉林省", "长春市", "朝阳区 南关区 宽城区 绿园区"),
    ("黑龙江省", "哈尔滨市", "南岗区 道里区 道外区 香坊区"),
    ("云南省", "昆明市", "五华区 盘龙区 官渡区 西山区"),
    ("贵州省", "贵阳市", "南明区 云岩区 观山湖区"),
    ("广西壮族自治区", "南宁市", "青秀区 兴宁区 西乡塘区 江南区"),
    ("山西省", "太原市", "小店区 迎泽区 杏花岭区 万柏林区"),
    ("甘肃省", "兰州市", "城关区 七里河区 安宁区"),
    ("海南省", "海口市", "龙华区 美兰区 秀英区"),
]
STREET_STEMS = ["人民", "解放", "建设", "中山", "和平", "胜利", "文化", "幸福", "光明", "朝阳",
                "长江", "黄河", "东风", "新华", "复兴", "团结", "青年", "迎宾", "滨河", "科技",
                "学府", "工业", "体育", "公园", "南京", "北京", "延安", "江南", "望江", "紫荆",
                "梧桐", "银杏", "枫林", "翠竹", "金桥", "长安", "春晖", "丰收", "振兴", "园林"]
STREET_SUFFIX = ["路", "街", "大道", "东路", "西路", "南路", "北路", "中路", "巷", "大街"]
COMMUNITIES = ["阳光花园", "翠湖小区", "锦绣家园", "和谐家园", "金色港湾", "碧水苑", "怡景新村",
               "华府名邸", "绿城小区", "东方明珠苑", "幸福里", "书香门第", "丽景湾", "新世纪花园"]
STATIONS = ["人民广场", "火车站", "体育中心", "市民中心", "大学城", "会展中心", "科技园", "西客站"]

FW = str.maketrans("0123456789", "０１２３４５６７８９")


# ---------------------------------------------------------------- fillers
def fillers(rng):
    """One fresh set of filler values; templates use whatever subset they need."""
    floor = rng.choice([rng.randint(1, 6), rng.randint(1, 18), rng.randint(1, 33)])
    room = f"{floor}{rng.randint(1, 12):02d}"
    floor2 = rng.randint(1, 26)
    room2 = f"{floor2}{rng.randint(1, 8):02d}"
    bld = rng.randint(1, 36)
    bld2 = rng.choice([b for b in range(1, 37) if b != bld])
    prov, city, dists = rng.choice(PROV_CITY)
    street = rng.choice(STREET_STEMS) + rng.choice(STREET_SUFFIX)
    tail = rng.choice([
        "", "", "",
        f"{rng.choice(COMMUNITIES)}{bld}栋{room}",
        f"{bld}号楼{rng.randint(1, 6)}单元{room}",
        f"{rng.choice(COMMUNITIES)}{bld}幢{room}室",
    ])
    use_prov = prov and rng.random() < 0.6
    addr = (prov if use_prov else "") + city + rng.choice(dists.split()) + street + f"{rng.randint(1, 399)}号" + tail
    hour = rng.randint(1, 11)
    return {
        "room": room, "room2": room2,
        "door": str(rng.choice([rng.randint(1, 30), rng.randint(1, 120), rng.randint(1, 299)])),
        "bld": str(bld), "bld2": str(bld2), "unit": str(rng.randint(1, 6)),
        "dorm": f"{rng.randint(1, 6)}{rng.randint(1, 40):02d}",
        "yard": str(rng.randint(1, 48)), "stall": str(rng.randint(1, 160)),
        "sur": rng.choice(SURNAMES), "sur2": rng.choice(SURNAMES),
        "fruit": rng.choice(FRUITS), "color": rng.choice(COLORS),
        "comm": rng.choice(COMMUNITIES), "station": rng.choice(STATIONS),
        "addr": addr,
        # non-address numbers
        "price": str(rng.choice([rng.randint(9, 99), rng.randint(100, 999), rng.randint(1000, 8999)])),
        "price3": str(rng.randint(120, 899)),
        "year": str(rng.randint(1988, 2025)), "grade": str(rng.randint(2015, 2025)),
        "month": str(rng.randint(1, 12)), "day": str(rng.randint(1, 31)),
        "hour": str(hour), "am": str(rng.randint(8, 11)), "pm": str(rng.randint(2, 5)), "hour24": str(rng.randint(8, 22)), "min": f"{rng.choice([0, 10, 15, 20, 30, 40, 45, 50]):02d}",
        "pop": str(rng.randint(800, 9800)), "cnt": str(rng.randint(3, 60)),
        "order": str(rng.randint(2000000, 9999999)) + rng.choice(["", str(rng.randint(10, 99))]),
        "jersey": str(rng.randint(1, 99)), "car": str(rng.randint(1, 16)), "row": str(rng.randint(1, 20)),
        "seatl": rng.choice("ABCDF"), "page": str(rng.randint(3, 320)), "q": str(rng.randint(1, 30)),
        "s1": str(rng.randint(0, 5)), "s2": str(rng.randint(0, 5)),
        "ver": f"{rng.randint(1, 9)}.{rng.randint(0, 12)}.{rng.randint(0, 9)}",
        "bus": str(rng.choice([rng.randint(1, 30), rng.randint(100, 399)])),
        "cap": str(rng.choice([8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 120])),
        "model": str(rng.randint(100, 9999)), "pct": str(rng.randint(3, 97)),
        "rank": str(rng.randint(1, 300)), "age": str(rng.randint(6, 88)),
        "lvl": str(rng.randint(3, 120)), "line": str(rng.randint(1, 18)),
        "num": str(rng.randint(1, 40)), "http": rng.choice(["502", "503", "404", "500", "302", "403"]),
        "score": str(rng.randint(380, 690)), "kg": str(rng.randint(42, 98)),
        "qn": f"{rng.choice('ABC')}{rng.randint(1, 199):03d}", "bat": rng.choice(["5", "7"]),
        "shoe": str(rng.randint(35, 45)), "issue": str(rng.randint(2, 360)),
        "gb": rng.choice(["8", "12", "16", "24"]), "train": str(rng.randint(100, 1999)),
        "mins": str(rng.randint(5, 90)), "ch": str(rng.randint(3, 480)), "sec": f"{rng.randint(0, 59):02d}",
        "temp": str(rng.randint(-8, 38)), "fee": str(rng.randint(3, 15)), "h": str(rng.randint(2, 9)),
        "flight": str(rng.randint(1000, 8999)), "bldcnt": str(rng.randint(6, 48)),
        "floors": str(rng.randint(5, 33)), "km": str(rng.randint(3, 42)), "cls": str(rng.randint(28, 56)),
        "seat": str(rng.randint(1, 45)), "col": str(rng.randint(1, 24)),
        "code": f"{rng.randint(1, 30)}-{rng.randint(1, 6)}-{rng.randint(1000, 9999)}",
        "r": str(rng.randint(1, 4)), "ting": str(rng.randint(1, 2)), "area": str(rng.randint(48, 160)),
        "wan": str(rng.randint(80, 860)), "presses": str(rng.randint(3, 9)), "mm": str(rng.randint(10, 59)),
        "zhe": rng.choice(["五", "六", "七", "八", "8.5", "9", "7.5"]),
        "body": rng.choice(["手指", "大拇指", "两根手指", "指甲", "手心"]),
        "nb": rng.choice(["两", "四", "六", "八"]),
        "dev": rng.choice(["遥控器", "门铃", "挂钟", "鼠标", "电子秤", "手电筒"]), "cm": str(rng.randint(150, 192)),
    }


# ---------------------------------------------------------------- templates
# Positives: the location of a person / delivery target is only a number
# (plus attached unit characters such as 号/室/号楼/栋/幢/单元/号院).
POS = [
    ("P01_live_room", "我住[[{room}]]，到了直接上来敲门就行。"),
    ("P02_roomno_is", "房间号是[[{room}]]，前台说刷卡就能上去。"),
    ("P03_dorm_prefix", "[[{dorm}]]宿舍的同学，晚上十点前把水卡还回来哈。"),
    ("P04_form_door", "登记信息｜户主：{sur}先生｜门牌：[[{door}号]]｜备注：家里有狗，按门铃。"),
    ("P05_moved_door_sp", "我搬到了 [[{door} 号]]，周末有空来坐坐呗。"),
    ("P06_door_reddoor", "就在 [[{door}号]]，{color}门那家，门口有棵石榴树。"),
    ("P07_bld_locker", "快递放[[{bld}号楼]]楼下的柜子就行，谢谢师傅。"),
    ("P08_go_dong", "明天下午去 [[{bld}栋]] 找{sur}老师拿资料。"),
    ("P09_dash_combo", "我家是[[{bld}-{room}]]，电梯坏了，麻烦走楼梯上来。"),
    ("P10_unit_room", "地址就写[[{unit}单元{room}]]，小区门卫都知道。"),
    ("P11_qa_door", "A：你家几号？B：[[{door}号]]，铁门刷了{color}漆的那个。"),
    ("P12_qa_room", "问：送到哪个房间？答：[[{room}]]。"),
    ("P13_takeout_door", "外卖小哥你好，放[[{room}]]门口就行，别敲门，孩子在睡觉。"),
    ("P14_yard", "我在[[{yard}号院]]，进胡同左手第三个门就是。"),
    ("P15_stall", "摊位在 [[{stall}号]]，卖酱菜的就是我，来了喊一声。"),
    ("P16_dong_unit_ask", "已经到楼下了，[[{bld}栋{unit}单元]]是哪个门啊？"),
    ("P17_forum_upstairs", "求助帖：楼主住[[{bld}号楼{room}]]，楼上天天半夜拖椅子，有同样遭遇的吗？"),
    ("P18_cs_roomno", "客服：请问您的房号是多少？用户：[[{room}]]，卫生间水管漏水了。"),
    ("P19_dorm_welcome", "我们宿舍是[[{dorm}]]，欢迎学弟学妹来串门。"),
    ("P20_newdoor_party", "新家门牌[[{door}号]]，乔迁宴定在周六中午，都来啊。"),
    ("P21_pickup_change", "取件地址改成[[{bld}栋{room}]]吧，之前那个已经不住了。"),
    ("P22_three_dash", "备注：[[{bld}-{unit}-{room}]]，放门口鞋柜上，谢谢。"),
    ("P23_shi_suffix", "麻烦师傅送到[[{room}室]]，出电梯右转第二间。"),
    ("P24_village_door", "住址一栏我只填了[[{door}号]]，村里人都认识我家。"),
    ("P25_upstairs_room", "我就住你楼上，[[{room}]]，有事喊一声就行。"),
    ("P26_fetch_key", "下班顺路的话去[[{bld}号楼]]帮我拿一下钥匙，我妈在家。"),
    ("P27_qinshi", "他现在住[[{dorm}]]寝室，你直接上去找他吧。"),
    ("P28_company_dong", "我们公司在[[{bld}栋]]，进园区左转第二栋就是。"),
    ("P29_qa_dong", "A：你住哪栋？B：[[{bld}栋]]，最靠东边那栋。"),
    ("P30_qa_unit_room", "A：几单元几号？B：[[{unit}单元{room}]]，上来记得换鞋。"),
    ("P31_homevisit", "家访安排：{sur}同学家在[[{door}号]]，我们下午三点到。"),
    ("P32_duty_room", "这周轮到[[{room}]]倒垃圾，别忘了啊。"),
    ("P33_notice_unit", "物业通知：[[{bld}号楼{unit}单元]]明天上午停水，请住户提前储水。"),
    ("P34_neighbor_aunt", "楼下[[{room}]]的阿姨又来借梯子了，借了三回都没还。"),
    ("P35_stall_fruit", "我在[[{stall}号]]摊位，过来尝尝新到的{fruit}。"),
    ("P36_zhuang", "车停在路边，你到[[{bld}幢]]门口等我五分钟。"),
    ("P37_form_recipient", "收件人：{sur}女士；地址：[[{bld}栋{room}]]；放门卫室也可以。"),
    ("P38_new_full_combo", "新地址发你：[[{bld}号楼{unit}单元{room}]]，记得在软件里改一下。"),
    ("P39_hotel_room", "酒店给我们分到[[{room}]]，推开窗就是海，太值了！"),
    ("P40_office_moved", "{sur}老师的办公室搬到[[{room}]]了，交作业去那儿。"),
    ("P41_door_upstairs", "我住[[{door}号]]二楼，楼梯在院子里面，别走错。"),
    ("P42_where_room", "地址？就[[{room}]]啊，你上次来过的，忘啦？"),
    ("P43_rent_listing", "转租：[[{bld}栋]]朝南单间，拎包入住，有意私聊。"),
    ("P44_pickup_wait", "到了给我发消息，我下来接你，我在[[{bld}号楼]]。"),
    ("P45_old_place", "老地方，[[{door}号]]，晚上七点不见不散。"),
    ("P46_group_parcel", "邻居群：[[{room}]]的快递被谁拿错了？麻烦送回来，谢谢！"),
    ("P47_form_two", "入住登记｜楼栋：[[{bld}栋]]｜房号：[[{room}]]｜入住人数：两人"),
    ("P48_two_dong", "我在[[{bld}栋]]，她在[[{bld2}栋]]，中间就隔一个小花园。"),
    ("P49_repair", "上门维修预约：[[{unit}单元{room}室]]，空调不制冷，周末都在家。"),
    ("P50_qa_dash", "A：送哪儿？B：[[{bld}-{room}]]，放门口就行。"),
    ("P51_ask_guard_yard", "找不到路就问门口保安[[{yard}号院]]怎么走，他们都知道。"),
    ("P52_door_renamed", "地址改为[[{door}号]]了，原来那块旧门牌已经拆掉。"),
    ("P53_share_room", "我和{sur}姐合租，住[[{room}]]，有空过来吃火锅。"),
    ("P54_doorno_bare", "门牌号 [[{door}]]，院门没锁，直接推门进来。"),
    ("P55_live_dong_room", "现在住[[{bld}栋{room2}]]，比之前那间大一点。"),
    ("P56_qa_moved", "A：你们搬哪去了？B：就隔壁，[[{door}号]]。"),
    ("P57_dorm_roommate", "室友说他今晚不回[[{dorm}]]了，门别反锁。"),
    ("P58_courier_note", "快递备注：送[[{room}]]，家里没人就放隔壁{sur}阿姨那。"),
]

# Negatives: numbers that are not addresses. Several reuse 号/室/楼/栋/单元.
NEG = [
    ("N01_price_yuan", "这件外套才{price}元，比商场便宜一半。"),
    ("N02_price_symbol", "￥{price}，包邮，今晚下单明天就到。"),
    ("N03_year_grad", "我{year}年毕业，之后换了三份工作。"),
    ("N04_date_home", "{day}号回家，票已经买好了。"),
    ("N05_weekday_date", "下周三 {day} 号开家长会，别迟到啊。"),
    ("N06_time", "会议改到明天上午{am}点{min}分，大家记一下。"),
    ("N07_population", "村里人口{pop}人，其中六十岁以上的占三成。"),
    ("N08_people_count", "昨天聚会来了 {cnt} 个人，菜点少了。"),
    ("N09_order_no", "订单号 {order}，麻烦帮我查一下物流到哪了。"),
    ("N10_jersey", "他穿 {jersey} 号球衣，场上跑得最快的那个。"),
    ("N11_train_seat", "座位是{car}车厢{row}{seatl}，靠窗，挺好。"),
    ("N12_page_question", "这道题在第{page}页，第{q}题，我也不会。"),
    ("N13_score", "最后比分{s1}比{s2}，看得我心脏都受不了。"),
    ("N14_app_version", "App 更新到 {ver} 以后老是闪退，有人遇到吗？"),
    ("N15_bus_route", "坐 {bus} 路车到终点站，再走五分钟就到。"),
    ("N16_meeting_cap", "会议室能坐 {cap} 人，周会够用了。"),
    ("N17_model_no", "我的笔记本型号是 X{model}，用了五年还没坏。"),
    ("N18_percent", "这次通过率只有{pct}%，比去年低不少。"),
    ("N19_rank", "期末全年级排第{rank}名，比上次进步了。"),
    ("N20_age", "我今年{age}岁，身体还算硬朗。"),
    ("N21_game_floor", "这游戏我打到第{lvl}层就卡住了，boss 太难打。"),
    ("N22_metro_line", "坐{line}号线地铁，中间换乘一次就到。"),
    ("N23_holiday_date", "{day}号放假，公司提前一天下班。"),
    ("N24_contestant", "第 {num} 号选手请到后台准备。"),
    ("N25_http_code", "打开网页一直显示 {http}，服务器是不是挂了？"),
    ("N26_glue", "{body}被502胶水粘住了，{sur}师傅说用热水泡一泡，管用吗？"),
    ("N27_exam_score", "高考考了 {score} 分，够不够上一本线？"),
    ("N28_weight", "体重 {kg} 公斤，这个月瘦了两斤。"),
    ("N29_queue", "叫号叫到 {qn} 了，前面还有十几个人。"),
    ("N30_battery", "帮我买{nb}节{bat}号电池，{dev}没电了。"),
    ("N31_shoe_size", "鞋码{shoe}，这款偏大半码，建议拍小一码。"),
    ("N32_episode", "第{issue}期节目太好笑了，笑到肚子疼。"),
    ("N33_phone_spec", "这款手机{gb}G内存，{price}元能入手吗？"),
    ("N34_dining_car", "{car}号车厢是餐车，饿了可以去那边买盒饭。"),
    ("N35_train_delay", "K{train}次列车晚点 {mins} 分钟，大家别着急。"),
    ("N36_households_vote", "共{cnt}户参加了这次投票，赞成的占多数。"),
    ("N37_doc_pages", "这份文件一共 {page} 页，记得双面打印。"),
    ("N38_birthday", "{month}月{day}号是我生日，别忘了。"),
    ("N39_chapter", "第{ch}章写得太拖了，看不下去，弃了。"),
    ("N40_run_time", "这次五公里用时{mm}分{sec}秒，配速还行。"),
    ("N41_temperature", "明天气温 {temp} 度，出门记得多穿点。"),
    ("N42_parking_fee", "停车费每小时 {fee} 元，停了 {h} 小时。"),
    ("N43_flight", "航班 CA{flight} 延误了，登机时间待定。"),
    ("N44_draw_lot", "抽签抽到 {num} 号，第三个上场，好紧张。"),
    ("N45_bld_count", "这个小区一共 {bldcnt} 栋楼，绿化做得不错。"),
    ("N46_floor_count", "整栋楼一共{floors}层，居然没有电梯。"),
    ("N47_classroom_cap", "这间阶梯教室能容纳{cap}人，讲座放这里吧。"),
    ("N48_gather_time", "{day}号上午九点半集合，别睡过头。"),
    ("N49_stock", "库存还剩{cnt}件，卖完就下架了。"),
    ("N50_class_size", "我们班 {cls} 个人，男生比女生多。"),
    ("N51_distance", "今天跑了{km}公里，腿都软了。"),
    ("N52_exam_seat", "考试座位号 {seat}，进场前看清楚别坐错。"),
    ("N53_grade_cohort", "他是{grade}级的学长，现在在读研。"),
    ("N54_cinema_seat", "电影晚上{hour}点开场，我买的是 {row} 排 {col} 座。"),
    ("N55_doc_number", "第{num}号文件已经下发，各部门照办。"),
    ("N56_room_price", "{price3}元一晚的房间，性价比还挺高。"),
    ("N57_pickup_code", "取件码 {code}，到驿站报这个就行。"),
    ("N58_layout", "这套房子{r}室{ting}厅，{area}平米，采光很好。"),
    ("N59_elevator_press", "电梯按钮按了{presses}下都没反应，物业快来看看。"),
    ("N60_chart_rank", "这首歌在榜单上排第{rank}位，比上周升了不少。"),
    ("N61_size_height", "身高{cm}，穿 L 码还是 XL 码合适？"),
    ("N62_discount", "店庆全场{zhe}折，满{price}减{fee}，算下来挺划算。"),
    ("N63_room_count_hotel", "这家酒店有{rank}间客房，旺季基本订满。"),
    ("N64_line_date_mix", "{day}号那天{line}号线停运，记得提前出门。"),
    ("N65_qa_date", "A：你几号到？B：{day}号，下午的车。"),
]

# Mixed: one ordinary full address plus a confusable non-address number.
MIX = [
    ("M01_delivery_date", "收货地址：[[{addr}]]，请{day}号之前送到，谢谢。"),
    ("M02_metro", "我家在[[{addr}]]，坐{line}号线过来，出站再走{mins}分钟。"),
    ("M03_sender_fee", "寄件人地址[[{addr}]]，运费{fee}元已付。"),
    ("M04_office_move", "新公司地址：[[{addr}]]，{day}号正式搬过去，到时请大家吃饭。"),
    ("M05_year_moved", "{year}年我们搬到了[[{addr}]]，一住就是好多年。"),
    ("M06_qa_bus", "A：你家具体在哪？B：[[{addr}]]，坐{bus}路公交到终点站下车。"),
    ("M07_order_fix", "订单{order}的收货地址填错了，麻烦改成[[{addr}]]。"),
    ("M08_visit_time", "家访地址：[[{addr}]]；时间：{month}月{day}号下午{pm}点。"),
    ("M09_listing", "[[{addr}]]这套房，{r}室{ting}厅，挂牌{wan}万，有兴趣可以约看。"),
    ("M10_gather_count", "团建集合点在[[{addr}]]，人到齐{cnt}个就发车。"),
]

MARK = re.compile(r"\[\[(.+?)\]\]")


def render(template, fill):
    """Format a template and turn [[...]] markers into character offsets."""
    raw = template.format(**fill)
    text, ents, pos = [], [], 0
    out_len = 0
    for m in MARK.finditer(raw):
        pre = raw[pos:m.start()]
        text.append(pre)
        out_len += len(pre)
        span = m.group(1)
        ents.append({"start": out_len, "end": out_len + len(span), "type": "address", "text": span})
        text.append(span)
        out_len += len(span)
        pos = m.end()
    text.append(raw[pos:])
    return "".join(text), ents


def plan(templates, n, rng):
    """Spread n records as evenly as possible over templates, in shuffled order."""
    ids = [t for i in range(n) for t in [templates[i % len(templates)]]]
    rng.shuffle(ids)
    return ids


def build():
    rng = random.Random(SEED)
    jobs = [("pos", t) for t in plan(POS, N_POS, rng)]
    jobs += [("neg", t) for t in plan(NEG, N_NEG, rng)]
    jobs += [("mixed", t) for t in plan(MIX, N_MIX, rng)]
    rng.shuffle(jobs)
    seen, records = set(), []
    for kind, (pid, tpl) in jobs:
        for _ in range(200):
            text, ents = render(tpl, fillers(rng))
            # ~7% of pos/neg records use full-width digits (char-for-char, spans stay valid)
            if kind != "mixed" and rng.random() < 0.07:
                text = text.translate(FW)
                for e in ents:
                    e["text"] = e["text"].translate(FW)
            if text not in seen:
                break
        else:
            raise RuntimeError(f"could not produce a unique text for {pid}")
        seen.add(text)
        records.append({"kind": kind, "pattern": pid, "text": text, "entities": ents})
    for i, r in enumerate(records, 1):
        r["id"] = f"numaddr_{i:05d}"
    return [{k: r[k] for k in ("id", "kind", "pattern", "text", "entities")} for r in records]


def check(records):
    for r in records:
        t = r["text"]
        assert 10 <= len(t) <= 120 <= 150, (r["id"], len(t))
        assert "[[" not in t and "]]" not in t and "@" not in t, r["id"]
        assert not re.search(r"[0-9０-９]{11,}", t), r["id"]  # nothing phone/ID-like
        prev_end = -1
        for e in sorted(r["entities"], key=lambda e: e["start"]):
            assert t[e["start"]:e["end"]] == e["text"], r["id"]
            assert e["type"] == "address" and e["start"] < e["end"], r["id"]
            assert e["start"] >= prev_end, ("overlap", r["id"])
            prev_end = e["end"]
        if r["kind"] == "neg":
            assert not r["entities"], r["id"]
        else:
            assert r["entities"], r["id"]
    used = collections.Counter(r["pattern"] for r in records)
    pos_used = {p for p in used if p.startswith("P")}
    neg_used = {p for p in used if p.startswith("N")}
    assert len(pos_used) >= 45 and len(neg_used) >= 45, (len(pos_used), len(neg_used))
    assert len({r["text"] for r in records}) == len(records)
    return used, pos_used, neg_used


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "numaddr.jsonl"
    records = build()
    used, pos_used, neg_used = check(records)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
    kinds = collections.Counter(r["kind"] for r in records)
    print("records:", len(records), dict(sorted(kinds.items())))
    print("distinct pos templates:", len(pos_used), "neg templates:", len(neg_used),
          "mixed templates:", len({p for p in used if p.startswith("M")}))
    print("gold address spans:", sum(len(r["entities"]) for r in records))
    lens = [len(r["text"]) for r in records]
    print("text length min/max:", min(lens), max(lens))
    for p, c in sorted(used.items()):
        print(f"  {p}: {c}")
    with open(out, "rb") as f:
        print("sha256:", hashlib.sha256(f.read()).hexdigest())


if __name__ == "__main__":
    main()
