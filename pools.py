"""Entity pools for synthetic PII insertion, split deterministically into TRAIN / EVAL halves.

Every pool item is assigned to exactly one half by a stable hash, so the held-out eval set (E2)
never shares a surname-type item, given-name item, road stem, English name or minority name
with the training data. Full names are additionally de-duplicated by the builders.
"""
import hashlib, random

def _half(item: str) -> str:
    return "eval" if int(hashlib.sha256(("p2s|" + item).encode()).hexdigest(), 16) % 2 else "train"

def split(items):
    out = {"train": [], "eval": []}
    for it in dict.fromkeys(items):
        out[_half(it)].append(it)
    return out

COMMON_SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤")
RARE_SURNAMES = list("晁庹邴禚亓綦蒯郗佘迟鞠甄昝笪逄蔺訾仉阚靳俞柏缪冼麦邝区蓝练栾郦禹隗巫乜檀臧闵满蒙乔聂单查")
COMPOUND_SURNAMES = ["欧阳", "司马", "上官", "诸葛", "东方", "皇甫", "尉迟", "公孙", "慕容", "令狐",
                     "长孙", "宇文", "夏侯", "司徒", "端木", "南宫", "独孤", "西门", "百里", "呼延",
                     "轩辕", "钟离", "闻人", "濮阳", "澹台", "公冶", "太史", "申屠", "赫连", "拓跋"]
GIVEN_CHARS = list("伟芳娜敏静丽强磊军洋勇艳杰涛明超秀霞平刚桂英华玉兰萍红建国志文斌辉鹏飞宇浩然子涵梓轩"
                   "欣怡诗雨晨思远博文嘉怡俊熙一诺可馨若曦景行昊天雅琪紫萱奕辰梦瑶书言沐阳亦舒知夏"
                   "晓峰永福德胜春生桂芝凤琴海燕雪梅丹丹少华卫东秋菊冬梅庆丰晓东建平立新红霞志刚")
NICK_PREFIX = ["小", "老", "阿"]
MINORITY_NAMES = ["扎西多吉", "格桑卓玛", "次仁央宗", "达瓦顿珠", "阿依古丽·吐尔逊", "买买提·艾力",
                  "古丽娜尔·阿不都", "热依拉·艾买提", "巴特尔", "其其格", "乌兰图雅", "苏和巴特尔",
                  "阿卜杜拉·买买提", "帕提古丽", "迪丽努尔·阿吉", "索朗旺姆", "旦增尼玛", "白玛措",
                  "宝音朝克图", "娜仁花", "额尔敦", "吉日嘎拉", "阿力木江", "努尔兰·别克", "其美多吉",
                  "央金拉姆", "洛桑丹增", "嘎玛次仁", "那顺乌日图", "哈斯巴根"]
EN_FIRST = ["Kevin", "Emily", "Jason", "Grace", "Leo", "Vivian", "Eric", "Cindy", "Tony", "Iris",
            "Frank", "Sophia", "Allen", "Coco", "Derek", "Wendy", "Ryan", "Joyce", "Ivan", "Amber",
            "Sean", "Tina", "Henry", "Olivia", "Peter", "Stella", "Bruce", "Yuki", "Calvin", "Nina"]
PINYIN_SURNAMES = ["Zhang", "Wang", "Li", "Liu", "Chen", "Zhao", "Huang", "Zhou", "Wu", "Xu",
                   "Sun", "Hu", "Zhu", "Gao", "Lin", "He", "Guo", "Ma", "Luo", "Liang",
                   "Song", "Zheng", "Xie", "Han", "Tang", "Feng", "Yu", "Dong", "Xiao", "Cheng"]
PINYIN_GIVEN = ["Wei", "Fang", "Na", "Min", "Jing", "Qiang", "Lei", "Jun", "Yang", "Yong",
                "Jie", "Tao", "Ming", "Chao", "Xiuying", "Xia", "Ping", "Gang", "Hao", "Yuxuan",
                "Zihan", "Xinyi", "Siyuan", "Haoran", "Yiran", "Jiaqi", "Mengyao", "Junjie", "Shuang", "Kai"]

PROVINCE_CITY = [("广东", "佛山", ["顺德区", "南海区", "禅城区"]), ("浙江", "绍兴", ["柯桥区", "上虞区", "诸暨市"]),
                 ("江苏", "徐州", ["铜山区", "鼓楼区", "沛县"]), ("四川", "绵阳", ["涪城区", "游仙区", "江油市"]),
                 ("湖南", "岳阳", ["岳阳楼区", "汨罗市", "华容县"]), ("山东", "潍坊", ["奎文区", "寿光市", "青州市"]),
                 ("河南", "洛阳", ["西工区", "涧西区", "偃师区"]), ("湖北", "宜昌", ["西陵区", "夷陵区", "当阳市"]),
                 ("福建", "泉州", ["丰泽区", "晋江市", "南安市"]), ("安徽", "芜湖", ["镜湖区", "鸠江区", "无为市"]),
                 ("江西", "赣州", ["章贡区", "南康区", "瑞金市"]), ("广西", "柳州", ["城中区", "鱼峰区", "柳江区"]),
                 ("云南", "曲靖", ["麒麟区", "宣威市", "陆良县"]), ("河北", "保定", ["竞秀区", "莲池区", "涿州市"]),
                 ("陕西", "宝鸡", ["渭滨区", "金台区", "岐山县"]), ("辽宁", "鞍山", ["铁东区", "铁西区", "海城市"]),
                 ("黑龙江", "齐齐哈尔", ["龙沙区", "建华区", "讷河市"]), ("吉林", "吉林", ["船营区", "昌邑区", "蛟河市"]),
                 ("山西", "大同", ["平城区", "云冈区", "灵丘县"]), ("贵州", "遵义", ["红花岗区", "汇川区", "仁怀市"])]
MUNICIPALITIES = [("上海", ["浦东新区", "闵行区", "徐汇区", "杨浦区"]), ("北京", ["朝阳区", "海淀区", "丰台区", "通州区"]),
                  ("天津", ["南开区", "河西区", "滨海新区"]), ("重庆", ["渝北区", "九龙坡区", "沙坪坝区"])]
ROAD_STEMS = ["人民", "解放", "建设", "中山", "和平", "胜利", "文化", "新华", "长江", "黄河", "光明", "幸福",
              "朝阳", "东风", "迎宾", "滨江", "青年", "工业", "学府", "花园", "振兴", "前进", "健康", "复兴",
              "金桥", "银河", "翠湖", "枫林", "桃源", "龙华", "望江", "沿河", "凤凰", "丽水", "锦绣", "春晓"]
ROAD_SUFFIX = ["路", "大道", "街", "巷", "北路", "南路", "东路", "西路"]
ESTATE_STEMS = ["阳光", "碧桂", "万科", "金色", "绿城", "华润", "保利", "翡翠", "御景", "世纪", "盛世", "名都",
                "锦江", "明珠", "雅居", "康城", "书香", "天悦", "领秀", "清水湾", "梧桐", "海棠", "香樟", "水岸"]
ESTATE_SUFFIX = ["小区", "花园", "家园", "苑", "公寓", "新村", "城", "府"]
VILLAGE_STEMS = ["王庄", "李家", "张湾", "刘集", "陈岗", "杨柳", "石桥", "大坪", "下河", "后坝", "南岭", "柏树",
                 "新屯", "白沙", "长岭", "黄泥", "马家", "沙河", "高坎", "东坡"]
EN_STREETS = ["Jinqiao Rd", "Huaihai Middle Rd", "Zhongshan Ave", "Binjiang Blvd", "Xueyuan Rd", "Keji Rd",
              "Renmin South Rd", "Jiefang Rd", "Wenhua St", "Hubin Rd", "Longhua Rd", "Taoyuan St"]
EMAIL_DOMAINS = ["qq.com", "163.com", "126.com", "gmail.com", "outlook.com", "foxmail.com", "sina.com", "hotmail.com"]

POOLS = {k: split(v) for k, v in {
    "rare_surname": RARE_SURNAMES, "compound_surname": COMPOUND_SURNAMES, "given": GIVEN_CHARS,
    "minority": MINORITY_NAMES, "en_first": EN_FIRST, "pinyin_surname": PINYIN_SURNAMES,
    "pinyin_given": PINYIN_GIVEN, "road": ROAD_STEMS, "estate": ESTATE_STEMS, "village": VILLAGE_STEMS,
    "en_street": EN_STREETS,
}.items()}
# common surnames are shared by both halves (they are the bulk of real names); full names are de-duplicated.
POOLS["common_surname"] = {"train": COMMON_SURNAMES, "eval": COMMON_SURNAMES}


class Gen:
    """Entity generator bound to one half ("train" or "eval")."""
    def __init__(self, half, seed, bldg_p=0.0):
        self.h = half; self.r = random.Random(seed); self.bldg_p = bldg_p  # building/unit/room suffix probability (released recipe: 0.5)
    def p(self, key):
        return self.r.choice(POOLS[key][self.h])
    def given(self):
        n = self.r.choice([1, 2, 2, 2])
        return "".join(self.p("given") for _ in range(n))
    def zh_name(self):
        k = self.r.random()
        if k < 0.55: return self.p("common_surname") + self.given()
        if k < 0.75: return self.p("rare_surname") + self.given()
        if k < 0.90: return self.p("compound_surname") + self.given()
        return self.p("minority")
    def nickname(self):
        s = self.p("common_surname"); pre = self.r.choice(NICK_PREFIX)
        return pre + (s if pre != "阿" else self.p("given"))
    def en_name(self):
        if self.r.random() < 0.5:
            return f"{self.p('en_first')} {self.p('pinyin_surname')}"
        return f"{self.p('pinyin_surname')} {self.p('pinyin_given')}"
    def phone(self):
        return self.r.choice(["13", "15", "17", "18", "19"]) + "".join(str(self.r.randint(0, 9)) for _ in range(9))
    def email(self):
        u = self.r.choice([self.p("pinyin_surname").lower() + self.p("pinyin_given").lower(),
                           self.p("en_first").lower() + str(self.r.randint(1, 9999))])
        return u + "@" + self.r.choice(EMAIL_DOMAINS)
    def national_id(self):
        area = str(self.r.randint(110000, 659000)); y = self.r.randint(1950, 2006)
        body = f"{area}{y}{self.r.randint(1, 12):02d}{self.r.randint(1, 28):02d}{self.r.randint(0, 999):03d}"
        w = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        return body + "10X98765432"[sum(int(a) * b for a, b in zip(body, w)) % 11]
    def bank_card(self):
        digits = [int(c) for c in "62" + "".join(str(self.r.randint(0, 9)) for _ in range(13))]
        tot = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 0: d = d * 2 - 9 if d * 2 > 9 else d * 2
            tot += d
        return "".join(map(str, digits)) + str((10 - tot % 10) % 10)
    def address(self):
        k = self.r.random()
        if k < 0.30:  # province-city-district-road-number
            pv, ct, ds = self.r.choice(PROVINCE_CITY)
            base = f"{pv}省{ct}市{self.r.choice(ds)}{self.p('road')}{self.r.choice(ROAD_SUFFIX)}{self.r.randint(1, 999)}号"
            if self.bldg_p and self.r.random() < self.bldg_p:
                base += self.r.choice([f"{self.r.randint(1, 30)}栋{self.r.randint(100, 3300)}室",
                                       f"{self.r.randint(1, 30)}栋{self.r.randint(1, 6)}单元{self.r.randint(101, 3304)}室",
                                       f"{self.r.randint(1, 20)}号楼{self.r.randint(101, 2804)}"])
            return base
        if k < 0.55:  # municipality estate building unit room
            m, ds = self.r.choice(MUNICIPALITIES)
            return (f"{m}市{self.r.choice(ds)}{self.p('estate')}{self.r.choice(ESTATE_SUFFIX)}"
                    f"{self.r.randint(1, 30)}栋{self.r.randint(1, 6)}单元{self.r.randint(1, 33)}{self.r.randint(1, 4):02d}室")
        if k < 0.70:  # rural
            pv, ct, ds = self.r.choice(PROVINCE_CITY)
            base = f"{ct}市{self.r.choice(ds)}{self.p('village')}村{self.r.randint(1, 12)}组{self.r.randint(1, 80)}号"
            if self.bldg_p and self.r.random() < self.bldg_p / 2:
                base += f"{self.r.randint(1, 3)}楼"
            return base
        if k < 0.85:  # intersection landmark
            a, b = self.p("road"), self.p("road")
            return (f"{self.p('road')}路与{b}街交叉口东北角{self.p('estate')}{self.r.choice(ESTATE_SUFFIX)}"
                    f"{self.r.randint(1, 20)}号楼")
        return f"Room {self.r.randint(1, 30)}{self.r.randint(1, 9):02d}, No. {self.r.randint(1, 500)} {self.p('en_street')}, {self.r.choice(['Shanghai', 'Shenzhen', 'Hangzhou', 'Chengdu', 'Suzhou'])}"
