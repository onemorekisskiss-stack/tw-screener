#!/usr/bin/env python3
"""完全免費版：只用證交所、櫃買中心的公開資料，產生可匯入網頁篩選器的 CSV。

資料來源
  行情（可指定日期，一天一次抓全市場）
    上市  https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX
    上櫃  https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes
  三大法人
    上市  https://www.twse.com.tw/rwd/zh/fund/T86
    上櫃  https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade
  本益比、殖利率、月營收年增（OpenAPI，只有最新一期）
    https://openapi.twse.com.tw/v1/ 與 https://www.tpex.org.tw/openapi/v1/

用法
  python build_free_csv.py                 # 全市場，輸出 out/screener.csv
  python build_free_csv.py --universe universe.txt   # 只輸出名單內的股票
  python build_free_csv.py --selftest      # 不連網，用假資料驗證流程
"""
import argparse, datetime as dt, re, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd

TODAY = dt.date.today()
CACHE = Path("cache")
NEED_DAYS = 65  # 需要的交易日數（算 60 日漲幅要 61 筆）
INST_DAYS = 5
COLS = ["code", "name", "industry", "price", "ret20", "ret60", "above20", "above60", "rsi", "volRatio",
        "instNet5", "marginChg", "revYoy", "epsGrowth", "pe", "yld", "avgValue", "exFlag"]
CODE_RE = re.compile(r"^[1-9]\d{3}$")  # 4 碼、不以 0 開頭：普通股（排除 ETF、權證、特別股）
UA = {"User-Agent": "Mozilla/5.0 (personal research; tw-screener-free)"}

TWSE_PX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_PX = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"
TWSE_INST = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_INST = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
OPENAPI = {
    "twse_pe": "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
    "tpex_pe": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis",
    "twse_rev": "https://openapi.twse.com.tw/v1/opendata/t187ap05_L",
    "tpex_rev": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O",
}


class Blocked(Exception):
    pass


class Http:
    """每次請求前先等待，避免被網站封鎖；失敗回傳 None，被擋（403/429）則拋出 Blocked。"""

    def __init__(self, sleep=3.0):
        self.sleep = sleep

    def get_json(self, url, params=None):
        import requests
        for attempt in range(3):
            time.sleep(self.sleep)
            try:
                r = requests.get(url, params=params, headers=UA, timeout=45)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (403, 429):
                if attempt == 2:
                    raise Blocked(f"HTTP {r.status_code}：請求被網站限制")
                time.sleep(60 * (attempt + 1))
                continue
            if r.status_code != 200:
                time.sleep(5)
                continue
            try:
                return r.json()
            except ValueError:
                time.sleep(5)
        return None


# ---------- 解析：依欄位名稱找欄，不依賴欄位順序 ----------
def num(x):
    try:
        return float(str(x).replace(",", "").replace("+", "").strip())
    except ValueError:
        return np.nan


def tables(j):
    """同時支援 {"tables":[{fields,data}]}、{"fields":..,"data":..}、{"fields9":..,"data9":..} 三種格式。"""
    if not isinstance(j, dict):
        return
    for t in j.get("tables") or []:
        if t.get("fields") and t.get("data"):
            yield t["fields"], t["data"]
    for k, v in j.items():
        if k.startswith("fields") and isinstance(v, list) and isinstance(j.get("data" + k[6:]), list):
            yield v, j["data" + k[6:]]


def col(fields, *keys):
    for key in keys:
        for i, f in enumerate(fields):
            if key in re.sub(r"<[^>]+>", "", str(f)):
                return i
    return None


def clean_code(x):
    return str(x).strip().strip('="').strip()


def unit(field):
    return 1000 if "仟" in str(field) else 1


def parse_quotes(j, close_keys):
    rows = []
    for f, d in tables(j):
        ic, icl = col(f, "證券代號", "代號"), col(f, *close_keys)
        iv, im, inm = col(f, "成交股數"), col(f, "成交金額"), col(f, "證券名稱", "名稱")
        if None in (ic, icl, iv, im):
            continue
        for r in d:
            code = clean_code(r[ic])
            close = num(r[icl])
            if not CODE_RE.match(code) or not close > 0:
                continue
            rows.append({"code": code, "name": str(r[inm]).strip() if inm is not None else "", "close": close,
                         "volume": num(r[iv]) * unit(f[iv]), "value": num(r[im]) * unit(f[im])})
    return pd.DataFrame(rows, columns=["code", "name", "close", "volume", "value"]).drop_duplicates("code")


def parse_inst(j):
    out = {}
    for f, d in tables(j):
        ic, inet = col(f, "證券代號", "代號"), col(f, "三大法人買賣超")
        if ic is None or inet is None:
            continue
        for r in d:
            out[clean_code(r[ic])] = num(r[inet]) * unit(f[inet])
    return out


# ---------- 取得每日資料（有快取，可中斷續跑） ----------
def get_prices(http, d):
    CACHE.mkdir(exist_ok=True)
    tag = f"{d:%Y%m%d}"
    hol = CACHE / f"holiday_{tag}"
    if hol.exists():
        return "holiday", None
    parts = []
    for mk in ("twse", "tpex"):
        p = CACHE / f"px_{mk}_{tag}.pkl"
        if p.exists():
            parts.append(pd.read_pickle(p))
            continue
        if mk == "twse":
            j = http.get_json(TWSE_PX, {"date": tag, "type": "ALLBUT0999", "response": "json"})
        else:
            j = http.get_json(TPEX_PX, {"date": f"{d:%Y/%m/%d}", "type": "EW", "response": "json"})
        if j is None:
            return "error", None
        df = parse_quotes(j, ("收盤價",) if mk == "twse" else ("收盤",))
        if df.empty:
            if mk == "twse":
                if "沒有符合" in str(j.get("stat", "")) and d < TODAY:
                    hol.touch()
                    return "holiday", None
                return ("pending" if d == TODAY else "error"), None
            print(f"  警告：{tag} 上櫃沒有資料，這天先略過上櫃")
            continue
        df["market"] = mk
        df.to_pickle(p)
        parts.append(df)
    return "ok", pd.concat(parts, ignore_index=True)


def get_inst(http, d):
    tag = f"{d:%Y%m%d}"
    res = {}
    for mk in ("twse", "tpex"):
        p = CACHE / f"inst_{mk}_{tag}.pkl"
        if p.exists():
            res.update(pd.read_pickle(p))
            continue
        if mk == "twse":
            j = http.get_json(TWSE_INST, {"date": tag, "selectType": "ALLBUT0999", "response": "json"})
        else:
            j = http.get_json(TPEX_INST, {"type": "Daily", "sect": "EW", "date": f"{d:%Y/%m/%d}", "id": "", "response": "json"})
        m = parse_inst(j) if j else {}
        if m:
            pd.to_pickle(m, p)
            res.update(m)
        else:
            print(f"  警告：{tag} {mk} 法人資料讀不到（欄位或格式可能有變）")
    return res


def get_openapi(http, name):
    import json
    p = CACHE / f"openapi_{name}_{TODAY}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    j = http.get_json(OPENAPI[name])
    if not isinstance(j, list):
        print(f"  警告：OpenAPI {name} 讀取失敗，相關欄位會留空")
        return []
    p.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return j


def pick(rec, *subs):
    for k, v in rec.items():
        if any(s in k.lower() for s in subs):
            return v
    return None


def fundamentals(http):
    """回傳 {code: {pe, yld, revYoy, industry}}，皆為最新一期的快照。"""
    out = {}
    for name in ("twse_pe", "tpex_pe"):
        for r in get_openapi(http, name):
            code = clean_code(pick(r, "code") or "")
            if CODE_RE.match(code):
                pe = num(pick(r, "peratio", "priceearning"))
                out.setdefault(code, {}).update(pe=pe if pe > 0 else np.nan, yld=num(pick(r, "yield")))
    for name in ("twse_rev", "tpex_rev"):
        for r in get_openapi(http, name):
            code = clean_code(r.get("公司代號", ""))
            if CODE_RE.match(code):
                out.setdefault(code, {}).update(revYoy=num(pick(r, "去年同月增減")), industry=str(r.get("產業別", "")))
    return out


# ---------- 指標計算 ----------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def build(http, universe=None):
    days, errors, d = [], 0, TODAY
    for _ in range(160):
        if d.weekday() < 5:
            st, df = get_prices(http, d)
            if st == "ok":
                days.append((d, df))
                print(f"  行情 {d} OK（{len(days)}/{NEED_DAYS}）")
            elif st == "error":
                errors += 1
                print(f"  行情 {d} 讀取失敗，略過（稍後重跑會補）")
        if len(days) >= NEED_DAYS:
            break
        d -= dt.timedelta(days=1)
    days.reverse()
    if len(days) < 61:
        sys.exit(f"可用交易日只有 {len(days)} 天（需要至少 61 天）。請稍後重跑，已抓的資料會用快取。")
    inst = {d: get_inst(http, d) for d, _ in days[-INST_DAYS:]}
    fund = fundamentals(http)

    px = pd.concat([df.assign(date=d) for d, df in days], ignore_index=True)
    rows = []
    for code, g in px.groupby("code"):
        if universe and code not in universe:
            continue
        g = g.sort_values("date")
        if len(g) < 61 or g["date"].iloc[-1] != days[-1][0]:
            continue  # 資料不足或最新一天沒交易（停牌）
        c, v = g["close"].reset_index(drop=True), g["volume"].reset_index(drop=True)
        f = fund.get(code, {})
        net = [inst[d].get(code, 0.0) for d in inst if inst[d]]
        avgv = v.tail(20).mean()
        jump = (c / c.shift(1)).tail(60)
        rows.append({
            "code": code, "name": g["name"].iloc[-1], "industry": f.get("industry", ""), "price": c.iloc[-1],
            "ret20": (c.iloc[-1] / c.iloc[-21] - 1) * 100, "ret60": (c.iloc[-1] / c.iloc[-61] - 1) * 100,
            "above20": int(c.iloc[-1] > c.tail(20).mean()), "above60": int(c.iloc[-1] > c.tail(60).mean()),
            "rsi": rsi(c).iloc[-1], "volRatio": v.iloc[-1] / v.iloc[-21:-1].mean(),
            "instNet5": sum(net) / avgv * 100 if net and avgv else np.nan,
            "marginChg": np.nan, "revYoy": f.get("revYoy", np.nan), "epsGrowth": np.nan,
            "pe": f.get("pe", np.nan), "yld": f.get("yld", np.nan),
            "avgValue": g["value"].tail(20).mean() / 1e8,
            "exFlag": int(((jump < 0.895) | (jump > 1.105)).any()),  # 1＝近 60 日有疑似除權息的跳空，漲幅僅供參考
        })
    return pd.DataFrame(rows, columns=COLS)


# ---------- 離線自我測試：用假資料模擬官方回應格式 ----------
class FakeHttp:
    TW = [str(1100 + i) for i in range(30)] + ["0050"]
    TP = [str(3100 + i) for i in range(10)] + ["03001P"]

    def price(self, code, d):
        seed = sum(map(ord, code))
        steps = np.random.default_rng(seed).normal(0.0005, 0.015, 1000)
        return (20 + seed % 200) * float(np.exp(steps[: max(d.toordinal() - dt.date(2025, 1, 1).toordinal(), 1)].sum()))

    def vol(self, code, d):
        return int(np.random.default_rng(sum(map(ord, code)) + d.toordinal()).integers(10**6, 6 * 10**6))

    def get_json(self, url, params=None):
        params = params or {}
        if "openapi" in url:
            if "BWIBBU" in url:
                return [{"Date": "x", "Code": c, "Name": "n", "PEratio": "15.5", "DividendYield": "3.2", "PBratio": "2"} for c in self.TW]
            if "peratio" in url:
                return [{"SecuritiesCompanyCode": c, "CompanyName": "n", "PriceEarningRatio": "12", "YieldRatio": "4.1"} for c in self.TP]
            return [{"公司代號": c, "公司名稱": "n", "產業別": "半導體業", "營業收入-去年同月增減(%)": "21.3"} for c in self.TW + self.TP]
        ds = params.get("date", "").replace("/", "")
        d = dt.date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        if d.toordinal() % 23 == 0:  # 模擬休市日
            return {"stat": "很抱歉，沒有符合條件的資料!", "tables": []}
        if "MI_INDEX" in url:
            data = [[c, "測試" + c, f"{self.vol(c, d):,}", "1,234", f"{self.vol(c, d) * self.price(c, d):,.0f}", "1", "1", "1",
                     f"{self.price(c, d):,.2f}", "+", "0.5"] for c in self.TW]
            f = ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差"]
            return {"stat": "OK", "tables": [{"title": "指數", "fields": ["指數", "收盤指數"], "data": [["x", "1"]]},
                                             {"title": "每日收盤行情", "fields": f, "data": data}]}
        if "dailyQuotes" in url:
            f = ["代號", "名稱", "收盤", "漲跌", "開盤", "最高", "最低", "成交股數", "成交金額(元)", "成交筆數"]
            data = [[c, "櫃" + c, f"{self.price(c, d):.2f}", "0", "1", "1", "1", f"{self.vol(c, d):,}",
                     f"{self.vol(c, d) * self.price(c, d):,.0f}", "10"] for c in self.TP]
            return {"stat": "ok", "tables": [{"fields": f, "data": data}]}
        if "T86" in url:
            f = ["證券代號", "證券名稱", "外陸資買賣超股數(不含外資自營商)", "投信買賣超股數", "三大法人買賣超股數"]
            return {"stat": "OK", "fields": f, "data": [[c, "n", "1", "1", f"{self.vol(c, d) // 50:,}"] for c in self.TW]}
        f = ["代號", "名稱", "三大法人買賣超股數合計"]
        return {"tables": [{"fields": f, "data": [[c, "n", f"{-self.vol(c, d) // 80:,}"] for c in self.TP]}]}


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", help="只輸出這份名單內的代號（一行一檔）")
    ap.add_argument("--out", default="out/screener.csv")
    ap.add_argument("--sleep", type=float, default=3.0, help="每次請求前等待秒數，請勿調得太小")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        CACHE = Path(tempfile.mkdtemp())
        http = FakeHttp()
    else:
        http = Http(a.sleep)
    uni = None
    if a.universe:
        uni = {l.split("#")[0].strip() for l in Path(a.universe).read_text(encoding="utf-8").splitlines()} - {""}
    try:
        df = build(http, uni)
    except Blocked as e:
        sys.exit(f"{e}。已抓的資料有快取，過一段時間再重跑會接續。")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(a.out, index=False, na_rep="", encoding="utf-8")
    print(f"已輸出 {len(df)} 檔 → {a.out}")


if __name__ == "__main__":
    main()
