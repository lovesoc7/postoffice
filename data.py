import os
import re
import urllib.parse
import requests
from email.header import decode_header

import glob
import zipfile

import pandas as pd

import holidays

def _parse_filename(cd: str) -> str:
    if not cd:
        return ""

    # 1. RFC 5987 규격 (filename*=utf-8''...) 처리 (대소문자 무시, 탐욕 방지)
    match = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;]+)", cd, re.IGNORECASE)
    if match:
        return urllib.parse.unquote(match.group(1).strip('"\''))

    # 2. 일반 filename="..." 추출
    match = re.search(r'filename="?([^";]+)"?', cd, re.IGNORECASE)
    if match:
        raw_fname = match.group(1).strip()
        
        # 2-1. 만약 filename 내부에 %EC%B2%A8%EB%B6%80 처럼 % 인코딩이 되어있는 경우
        if '%' in raw_fname:
            return urllib.parse.unquote(raw_fname)

        # 2-2. Latin-1(ISO-8859-1)로 잘못 읽힌 바이너리 복원 시도
        try:
            raw_bytes = raw_fname.encode('latin1')
            
            # UTF-8 우선 시도
            try:
                return raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # 한국 구형 서버(CP949 / EUC-KR) 디코딩 시도
                return raw_bytes.decode('cp949')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # 2-3. RFC 2047 인코딩 형태 (=?UTF-8?B?...?=) 처리
        try:
            decoded_segments = decode_header(raw_fname)
            result = ""
            for text, encoding in decoded_segments:
                if isinstance(text, bytes):
                    result += text.decode(encoding or 'utf-8', errors='replace')
                else:
                    result += text
            return result
        except Exception:
            pass

        return raw_fname

    return ""

# 다운로드할 링크 목록
urls = [ # 2026. 6. x. ~ # 2025. 1. x. 
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100111731.do",# 2026. 6. x.
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100110895.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100110116.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100109269.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100108556.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100107988.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100106913.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100106196.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100105657.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100105651.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100104581.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100103810.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100103063.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100101992.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100100807.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100099927.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100099064.do",
    "https://www.koreapost.go.kr/common/proc/kpost/bbs/114/fileDownLoad/100098132.do",# 2025. 1. x.
]
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
all_regions = ['강원특별자치도',
               '경기도','경상남도','경상북도',
               '광주광역시','대구광역시','대전광역시','부산광역시',
               '서울특별시','세종특별자치시',
               '울산광역시','인천광역시',
               '전라남도','전라북도','전북특별자치도','제주특별자치도','충청남도','충청북도']
region_type = pd.CategoricalDtype(categories=all_regions, ordered=False)

# 행정구역 명칭 변경 이력 대응용 매핑
# - 강원도 -> 강원특별자치도 (2023.6.11)
# - 전라북도 -> 전북특별자치도 (2024.1.18)
# 과거 연도 데이터를 나중에 합칠 경우를 대비해 구명칭이 섞여 들어와도
# 안전하게 표준 명칭으로 통일한다 (raw 데이터가 신명칭이어도 그대로 통과됨).
_legacy_region_map = {
    '강원도': '강원특별자치도',
    '전라북도': '전북특별자치도',
    '제주도': '제주특별자치도',
}

def _normalize_region_names(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["접수지역", "배달지역"]:
        if col in df.columns:
            # 구명칭('강원도', '전라북도')이 실제 데이터에 존재할 때만 컬럼 변경 진행
            if df[col].isin(_legacy_region_map.keys()).any():
                df[col] = df[col].replace(_legacy_region_map)
    return df

def _data_download(save_dir:str="./raw"):
    """
    1.물류데이터다운로드
    """
    os.makedirs(save_dir, exist_ok=True)

    for url in urls:
        res = requests.get(url, headers=headers, stream=True)
        
        if res.status_code == 200:
            # Content-Disposition 응답 헤더에서 한글 파일명 추출
            cd = res.headers.get("Content-Disposition")
            filename = None

            # 파일명 추출
            if cd:
                filename = _parse_filename(cd)
            
            # 헤더에 파일명이 없는 비상 상황용 예비 파일명
            if not filename:
                file_id = url.split("/")[-1].replace(".do", "")
                filename = f"downloaded_file_{file_id}.csv"
                
            file_path = os.path.join(save_dir, filename)
            
            # 파일 쓰기
            with open(file_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"저장 완료: {filename}")
        else:
            print(f"다운로드 실패 (상태코드 {res.status_code}): {url}")

def _unzip_downloaded_data(src_path: str = "./raw", dst_path: str = "./unziped"):
    """
    2. 다운로드한 파일들 압축 풀기 (하위 디렉토리명을 파일명에 병합)
    """
    zip_files = glob.glob(f'{src_path}/*.zip')
    
    # 저장할 최상위 디렉토리 미리 생성
    os.makedirs(dst_path, exist_ok=True)
    
    for z_file in zip_files:
        print(f"압축 해제 중: {z_file}")
        with zipfile.ZipFile(z_file, 'r') as zip_file:
            for member in zip_file.infolist():
                # 1. 디렉터리 자체 엔트리는 스킵
                if member.is_dir():
                    continue

                # 2. 파일명 인코딩 복원 (CP949 / UTF-8)
                try:
                    decoded_path = member.filename.encode('cp437').decode('cp949')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    try:
                        decoded_path = member.filename.encode('cp437').decode('utf-8')
                    except Exception:
                        decoded_path = member.filename

                # 3. 🔥 [핵심] 경로 구분자(/, \)를 언더바(_)로 바꿔서 파일명에 병합!
                # 예: "202602/강원도/소포.csv" -> "202602_강원도_소포.csv"
                flat_filename = decoded_path.replace('/', '_').replace('\\', '_').lstrip('_')

                if not flat_filename:
                    continue

                # 4. dst_path 바로 밑에 병합된 파일명으로 저장
                target_path = os.path.join(dst_path, flat_filename)

                with zip_file.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

def _to_pandas_table(src_path: str="./unziped")->pd.DataFrame:
    """
    3. csv데이터들을 하나의 pandas테이블로 합치기
    """
    # unziped 폴더 내 모든 csv 파일 경로 가져오기
    file_list = glob.glob(f'{src_path}/*.csv')

    # 파일들을 각각 읽어온 뒤 한 번에 세로 병합
    df_list = [pd.read_csv(file) for file in file_list]
    total_df = pd.concat(df_list, ignore_index=True) # 열이 일치하지 않으면 모두 살림..

    return total_df

def _preprocess_pandas_table(df:pd.DataFrame) -> pd.DataFrame:
    """
    4. pandas 테이블을 전처리함
    """
    # 접수일자를 문자형에서 날짜형으로 형변환
    df['접수일자'] = pd.to_datetime(df['접수일자'])

    # 행정구역 명칭 변경 대응 (전라북도->전북특별자치도 등) - 범주형 변환 전에 필수
    df = _normalize_region_names(df)

    # 접수지역, 배달지역을 범주형으로 변환

    df = df.dropna(subset=['접수지역'])
    df = df.dropna(subset=['배달지역'])
    # temp_st_region_list = df['접수지역'].unique().tolist()
    # temp_dst_region_list = df['배달지역'].unique().tolist()

    df['접수지역'] = df['접수지역'].astype(region_type)
    df['배달지역'] = df['배달지역'].astype(region_type)

    # ⚠️ 안전장치: 범주 목록에 없는 값이 있으면 조용히 사라지지 않고 반드시 알림
    n_bad_pickup = df['접수지역'].isna().sum()
    n_bad_deliver = df['배달지역'].isna().sum()
    if n_bad_pickup > 0 or n_bad_deliver > 0:
        raise ValueError(
            f"region_type 범주에 없는 지역명이 발견되어 {n_bad_pickup + n_bad_deliver}건이 "
            f"NaN으로 변환될 뻔했습니다 (접수지역 {n_bad_pickup}건, 배달지역 {n_bad_deliver}건). "
            f"data.py의 all_regions / _legacy_region_map을 원본 데이터 표기에 맞춰 업데이트하세요."
            # f"unique목록: st:{temp_st_region_list}, dst:{temp_dst_region_list}."
        )

    # 일자, 지역별로 그룹핑 (observed=True: 실제 관측된 조합만 집계, 불필요한 전체조합 확장 방지)
    df = df.groupby(['접수일자', '접수지역' , '배달지역'], observed=True).agg(
        총중량_kg=('중량', lambda x: x.sum() / 1000.0), # g -> kg 단위 변환
    ).reset_index()
    return df

def _add_date_features(df: pd.DataFrame, ) -> pd.DataFrame:
    """
    5. 예측에 도움이 되는 날짜피처들을 추가함
    """
    date_col: str = '접수일자'
    df = df.copy()
    df[date_col] = pd.to_datetime(date_col) if not pd.api.types.is_datetime64_any_dtype(df[date_col]) else df[date_col]
    
    # 0. merge_asof 연산을 위한 오름차순 정렬
    df = df.sort_values(date_col).reset_index(drop=True)

    # 1. 기본 날짜 피처
    df['연도'] = df[date_col].dt.year
    df['월'] = df[date_col].dt.month
    df['일'] = df[date_col].dt.day
    df['요일'] = df[date_col].dt.dayofweek  # 0:월, 1:화, ..., 5:토, 6:일 (주말 패턴 자동 학습)

    # 2. 법정 공휴일 여부만 순수하게 추출 (주말 제외!)
    years = df['연도'].unique().tolist()
    kr_holidays = holidays.KR(years=years)
    df['공휴일여부'] = df[date_col].isin(kr_holidays).astype(int)

    # 3. 공휴일 날짜만 추출 (D-Day 계산용)
    public_holiday_dates = pd.DataFrame({
        'holiday_date': df.loc[df['공휴일여부'] == 1, date_col].unique()
    }).sort_values('holiday_date')

    # 공휴일 데이터가 존재할 경우 D-Day 계산
    if not public_holiday_dates.empty:
        # [D-Minus] 다음 공휴일까지 남은 일수
        df = pd.merge_asof(
            df, public_holiday_dates,
            left_on=date_col, right_on='holiday_date',
            direction='forward'
        )
        df['다음_공휴일까지_남은일수'] = (df['holiday_date'] - df[date_col]).dt.days.fillna(999).astype(int)
        df.drop(columns=['holiday_date'], inplace=True)

        # [D-Plus] 지난 공휴일로부터 지난 일수
        df = pd.merge_asof(
            df, public_holiday_dates,
            left_on=date_col, right_on='holiday_date',
            direction='backward'
        )
        df['지난_공휴일부터_지난일수'] = (df[date_col] - df['holiday_date']).dt.days.fillna(999).astype(int)
        df.drop(columns=['holiday_date'], inplace=True)
    else:
        df['다음_공휴일까지_남은일수'] = 999
        df['지난_공휴일부터_지난일수'] = 999

    # 4. 공휴일 전후 물류 특수기 플래그
    df['공휴일_직전_3일내'] = df['다음_공휴일까지_남은일수'].between(1, 3).astype(int)
    df['공휴일_직후_2일내'] = df['지난_공휴일부터_지난일수'].between(1, 2).astype(int)

    return df

def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    6. 예측에 도움이 되는 지연피처들을 추가함
    """
    df = df.copy()
    
    # 1. 지역별 + 날짜순으로 정렬 (시계열 연산 필수 조건)
    group_cols = ['접수지역', '배달지역']
    df = df.sort_values(group_cols + ['접수일자']).reset_index(drop=True)

    # 2. Lag 피처 (과거 특정 시점 물량)
    # lag_1: 어제 물량, lag_7: 지난주 동요일 물량, lag_14: 2주 전 동요일 물량
    for lag in [1, 2, 7, 14]:
        df[f'lag_{lag}'] = df.groupby(group_cols)['총중량_kg'].shift(lag)

    # 3. Rolling 피처 (최근 N일간의 이동평균 및 표준편차)
    # shift(1)을 먼제 해줘서 오늘(t) 데이터가 계산에 들어가는 누수 방지!
    grouped_target = df.groupby(group_cols)['총중량_kg']
    
    # 최근 7일 평균 / 표준편차
    df['rolling_7_mean'] = grouped_target.transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_7_std']  = grouped_target.transform(lambda x: x.shift(1).rolling(7).std())
    
    # 최근 14일 평균
    df['rolling_14_mean'] = grouped_target.transform(lambda x: x.shift(1).rolling(14).mean())

    # 4. 파생 비율 피처 (어제 물량이 최근 7일 평균 대비 얼마나 튀었는지)
    # 1보다 크면 최근 평균보다 물량이 늘어나는 상승세
    df['ratio_lag1_to_roll7'] = df['lag_1'] / (df['rolling_7_mean'] + 1e-5)

    return df

#######################################################################################################

def print_urls():
    print(urls)
def download_pipeline(save_dir:str="./raw",zip_path: str = "./unziped"):
    """
    최종 데이터 다운로드 파이프라인: urls리스트에 데이터들을 다운로드후, 압축을 해제함
    save_dir: 저장할 위치
    zip_path: 압축해제할 위치
    """
    _data_download(save_dir)
    _unzip_downloaded_data(src_path=save_dir, dst_path=zip_path)
def preprocess_pipeline(src_path: str="./unziped") -> pd.DataFrame:
    """
    최종 전처리 파이프라인: 파일들을 읽어서, 하나의 데이터프레임으로 합치고, 전처리함
    src_path: 파일들이 위치한 경로
    """
    df = _to_pandas_table(src_path)
    print("preprocess_pipeline: _to_pandas_table done..")
    df = _preprocess_pandas_table(df)
    print("preprocess_pipeline: _preprocess_pandas_table done..")
    df = _add_date_features(df)
    print("preprocess_pipeline: _add_date_features done..")
    df = _add_lag_and_rolling_features(df)
    print("preprocess_pipeline: _add_lag_and_rolling_features done..")
    df = df.dropna().reset_index(drop=True) # 지연피처 추가시에 생긴 nan제거
    print("preprocess_pipeline: all process done..")
    return df
def split_timeseries_data(
    df: pd.DataFrame, 
    date_col: str = '접수일자', 
    test_days: int = 30,  # 최근 N일을 Test(검증) 세트로 사용
)->tuple[pd.DataFrame, pd.DataFrame]:
    """
    최대 날짜기준 test_days이전날짜를 기준으로 쪼갬
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 분할 기준일 계산
    max_date = df[date_col].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)

    # 3. 과거(Train)와 미래(Test) 분할
    train_mask = df[date_col] <= cutoff_date
    test_mask = df[date_col] > cutoff_date

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # 6. 분할 결과 정보 출력
    print("=" * 50)
    print("📊 시계열 데이터셋 분할 완료")
    print("=" * 50)
    print(f" 전체 데이터 기간: {df[date_col].min().strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    print(f" Train 데이터   : {train_df[date_col].min().strftime('%Y-%m-%d')} ~ {train_df[date_col].max().strftime('%Y-%m-%d')} ({len(train_df):,}건)")
    print(f" Test 데이터    : {test_df[date_col].min().strftime('%Y-%m-%d')} ~ {test_df[date_col].max().strftime('%Y-%m-%d')} ({len(test_df):,}건)")
    print("=" * 50)

    return train_df, test_df