#!/usr/bin/env python
# coding: utf-8

# In[15]:


import streamlit as st
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError  # SQLAlchemyError 임포트
from sqlalchemy.sql import update, text, delete  # update, text 함수 가져오기
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from scipy.stats import norm


# In[16]:


# MariaDB 연결 설정
db_config = {
    'user': 'root',
    'password': 'shimsang12',
    'host': 'localhost',
    'port': '3306',
    'database': 'sh'
}


# In[17]:


# SQLAlchemy 엔진 생성
DATABASE_URL = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"


# In[18]:


engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True
)


# In[19]:


# 세션 생성기
Session = sessionmaker(bind=engine)


# In[20]:


def load_data():
    """데이터베이스에서 데이터를 로드하는 함수"""
    session = Session()
    try:
        # 데이터베이스에서 테이블 가져오기
        metadata = MetaData()
        table = Table('company', metadata, autoload_with=engine)

        # 쿼리 실행
        query = session.query(table)
        df = pd.read_sql(query.statement, session.bind)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df

    except SQLAlchemyError as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환

    finally:
        session.close()


# In[21]:


def update_data():
    """데이터를 업데이트하는 함수"""
    st.write("update_data 함수 호출됨")  # 디버깅 로그 추가
    data = load_data()
    with st.form(key='update_form'):
        if not data.empty:
            # AgGrid 설정
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_pagination(paginationAutoPageSize=True)  # 페이지네이션 설정
            gb.configure_side_bar()  # 사이드바 설정
            gb.configure_selection('multiple', use_checkbox=True)  # 체크박스 설정

            # 모든 열을 편집 가능하게 설정
            editable_js = JsCode("""
            function(params) {
                return params.node.isSelected();
            }
            """)
            for col in data.columns:
                gb.configure_column(col, editable=editable_js)
                
            # 소수 입력을 위한 열 설정
            gb.configure_column('DIM_W', type='numericColumn', precision=2)
            gb.configure_column('DIM_H', type='numericColumn', precision=2)
            
            gridOptions = gb.build()

            # AgGrid 표시
            grid_response = AgGrid(
                data,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.VALUE_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                theme='streamlit',  # 테마 설정
                enable_enterprise_modules=True,
                height=500,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True  # unsafe_js_code 허용
            )

            # 편집된 데이터 가져오기
            edited_data = grid_response['data']
            st.subheader("편집된 데이터")
            st.write(edited_data)

            # 선택된 행 출력
            selected_rows = pd.DataFrame(grid_response['selected_rows'])
            if not selected_rows.empty:
                st.subheader("선택된 행")
                st.write(selected_rows)

            # 편집된 데이터를 MariaDB에 반영하는 함수
            def update_db(data):
                 # NaN 값을 None으로 변환
                data = data.where(pd.notnull(data), None)
                
                metadata = MetaData()
                table = Table('company', metadata, autoload_with=engine)
                try:
                    with engine.connect() as connection:
                        for index, row in data.iterrows():
                            #NaN 값을 None으로 변환한 후 확인
                            row = row.where(pd.notnull(row), None)
                            st.write(f"Updating row: {row.to_dict()}")  # 디버깅 로그 추가
                            
                            stmt = (
                                update(table)
                                .where(table.c.id == row['id'])
                                .values(
                                    company=row['company'],
                                    PART_NAME=row['PART_NAME'],
                                    Demand_ID=row['Demand_ID'],
                                    PART_NO=row['PART_NO'],
                                    QTY=row['QTY'],
                                    DATE=row['DATE'],
                                    DIM_W=row['DIM_W'],
                                    DIM_H=row['DIM_H']
                                )
                            )
                            st.write(f"Executing: {stmt}")  # 디버깅 로그 추가
                            connection.execute(stmt)
                        connection.commit()  # 트랜잭션 커밋
                except SQLAlchemyError as e:
                    st.error(f'데이터베이스 업데이트 중 오류가 발생했습니다: {e}')

            # 데이터베이스 업데이트 버튼
            if st.form_submit_button('데이터베이스 업데이트'):
                update_db(edited_data)
                st.success('데이터베이스가 성공적으로 업데이트되었습니다.')
                 # 세션 상태를 유지한 채로 페이지를 새로고침
                st.session_state['page'] = 'update_form'
                
        else:
            st.warning('데이터가 없습니다.')


# In[22]:


def add_data():
    """데이터를 추가하는 함수"""
    st.write("add_data 함수 호출됨")  # 디버깅 로그 추가
    
    with st.form(key='add_form'):
        company = st.text_input('Company')
        part_name = st.text_input('PART_NAME')
        demand_id = st.text_input('Demand_ID')
        part_no = st.text_input('PART_NO')
        qty = st.number_input('QTY', step=1, format='%d')  # 소수 자릿수를 0으로 설정
        date = st.date_input("Date")
        dim_W = st.number_input('DIM_W', step=1, format='%d')
        dim_H = st.number_input('DIM_H', step=1, format='%d')
        submit_button = st.form_submit_button(label='데이터 추가')

        if submit_button:
            try:
                with engine.begin() as connection:
                    sql = text("""
                    INSERT INTO company (company, PART_NAME, Demand_ID, PART_NO, QTY, DATE, DIM_W, DIM_H)
                    VALUES (:company, :part_name, :demand_id, :part_no, :qty, :date, :dim_W, :dim_H)
                    """)
                    values = {
                        'company': company,
                        'part_name': part_name,
                        'demand_id' : demand_id,
                        'part_no': part_no,
                        'qty': qty,
                        'date': date.strftime('%Y-%m-%d'),
                        'dim_W': dim_W,
                        'dim_H': dim_H
                    }
                    connection.execute(sql, values)
                    st.write('데이터 추가 완료!')
                    # 폼 제출 후 세션 상태 변경
                    st.session_state.form_submitted = True
            except Exception as e:
                st.write('오류:', e)            


# In[23]:


# 데이터 삭제 함수
def delete_data():
    """데이터를 삭제하는 함수"""
    st.write("delete_data 함수 호출됨")
    data = load_data()
    
    with st.form(key='delete_form'):
        if not data.empty:
            # AgGrid 설정
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_pagination(paginationAutoPageSize=True)  # 페이지네이션 설정
            gb.configure_side_bar()  # 사이드바 설정
            gb.configure_selection('multiple', use_checkbox=True)  # 체크박스 설정
            
            gridOptions = gb.build()
            
            # AgGrid 표시
            grid_response = AgGrid(
                data,
                gridOptions=gridOptions,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                theme='streamlit',  # 테마 설정
                enable_enterprise_modules=True,
                height=500,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True  # unsafe_js_code 허용
            )
            
            # 선택된 행 가져오기
            selected_rows = pd.DataFrame(grid_response['selected_rows'])
            if not selected_rows.empty:
                st.subheader("선택된 행")
                st.write(selected_rows)
            
            # 데이터베이스에서 선택된 행 삭제 함수
            def delete_db(data):
                metadata = MetaData()
                table = Table('company', metadata, autoload_with=engine)
                try:
                    with engine.connect() as connection:
                        for index, row in data.iterrows():
                            stmt = (
                                delete(table)
                                .where(table.c.id == row['id'])
                            )
                            st.write(f"Executing: {stmt}")  # 디버깅 로그 추가
                            connection.execute(stmt)
                        connection.commit()  # 트랜잭션 커밋
                except SQLAlchemyError as e:
                    st.error(f'데이터베이스 삭제 중 오류가 발생했습니다: {e}')
            
            # 데이터베이스에서 선택된 행 삭제
            if st.form_submit_button("선택된 행 삭제"):
                delete_db(selected_rows)
                st.success('선택된 행이 성공적으로 삭제되었습니다.')
                # 세션 상태를 유지한 채로 페이지를 새로고침
                st.session_state['page'] = 'delete_form'
                st.experimental_rerun()  # 강제로 새로고침
        else:
            st.warning('데이터가 없습니다.')


# In[24]:


def calculate_Z_bench():
    """공정능력구하는 함수"""
    st.write("process_analysis 함수 호출됨")
    data = load_data()
    part_no = st.selectbox("Part NO", data['PART_NO'].unique())
    
    if part_no:
        part_data = data[data['PART_NO'] == part_no]
        dimension = st.selectbox("Dimension", ['DIM_W', 'DIM_H'])
        
        if dimension:
            mean = part_data[dimension].mean()
            std_dev = part_data[dimension].std()
            
            with st.form(key='process_form'):
                usl = st.number_input("USL", value=0.0)
                lsl = st.number_input("LSL", value=0.0)
                submit_button = st.form_submit_button(label='Calculate Z_bench')
                
                if submit_button:
                    z_upper = (usl - mean) / std_dev
                    z_lower = (mean - lsl) / std_dev
                    
                    p_upper = 1 - norm.cdf(z_upper)
                    p_lower = 1 - norm.cdf(z_lower)
                    
                    p_total = p_upper + p_lower
                    z_bench = norm.ppf(1 - p_total)
                    
                    st.write(f"Z_bench: {z_bench}")
                    
                    # 불량률 계산
                    z_shift = 1.5  # 일반적으로 사용하는 Z shift 값
                    shifted_z = z_bench + z_shift
                    defect_rate = (1 - norm.cdf(shifted_z)) * 2  # 양쪽 꼬리의 합
                    
                    st.write(f"Z shift 적용 후 Z 값: {shifted_z}")
                    st.write(f"불량률: {defect_rate * 100:.6f}% (1백만 개당 {defect_rate * 1_000_000:.2f} 개)")


# In[25]:


def plot_time_series():
    st.write("시계열 그래프 함수 호출됨")
    data = load_data()
    
    # PART_NO 선택
    part_no = st.selectbox('Select PART_NO', data['PART_NO'].unique())
    
    # DATE 선택
    date_range = st.date_input('Select Date Range', [data['DATE'].min().date(), data['DATE'].max().date()])
    dimension = st.selectbox('Select Dimension', ['DIM_W', 'DIM_H'])
    
    # 선택된 PART_NO와 DATE 범위에 따른 데이터 필터링
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    filtered_data = data[(data['PART_NO'] == part_no) & (data['DATE'].between(start_date, end_date))]
    
    if not filtered_data.empty:
        # DATE를 인덱스로 설정하고 날짜 순서대로 정렬
        filtered_data = filtered_data.set_index('DATE').sort_index()
       
        # 시계열 그래프 그리기
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (14,14))
        
        ax1.plot(filtered_data.index, filtered_data[dimension], label=dimension, marker = 'o')
        ax1.set_title(f'Time Series for {dimension} - PART_NO: {part_no}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(dimension)
        ax1.legend()
        ax1.grid(True)          
                       
        # R 관리도 계산
        ranges = filtered_data[dimension].diff().dropna()
        R_bar = ranges.mean()
        sigma = ranges.std()
        UCL = R_bar + 3*sigma
        LCL = R_bar - 3*sigma
        
        # R 관리도 그리기
        ranges = filtered_data[dimension].diff().dropna()
        ax2.plot(filtered_data.index[1:], ranges, label='Range', marker='o')
        ax2.axhline(y=R_bar, color='green', linestyle='--', label='R-bar')
        ax2.axhline(y=UCL, color='red', linestyle='--', label='UCL')
        ax2.axhline(y=LCL, color='red', linestyle='--', label='LCL')
        ax2.set_title(f'R Control Chart for {dimension} - PART_NO: {part_no}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Range')
        ax2.legend()
        ax2.grid(True)
                        
        # 그래프 출력
        st.pyplot(fig)
    else:
        st.write('No data available for the selected PART_NO and Date range.')


# In[26]:


def reset_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state['page'] = "home"  # 초기 상태로 설정
    st.session_state['main_selectbox'] = "Menu를 선택하세요"  # 재고현황 selectbox 초기화
    st.session_state['update_selectbox'] = "Menu를 선택하세요"  # 데이터 수정 selectbox 초기화
    st.session_state['inventory_selectbox'] = "Menu를 선택하세요"  # 재고 관리 selectbox 초기화


# In[27]:


def main():
    st.sidebar.title('메뉴')
    
    menu = ["협력사별 재고현황", "Part NO 별 재고현황"]
    choice = st.sidebar.selectbox('재고현황', ["Menu를 선택하세요"] + menu, key='main_selectbox')
    
    inventory_update_menu = ["재고 업데이트", "재고 추가", "재고 삭제"]
    inventory_update_choice = st.sidebar.selectbox("재고수정", ["Menu를 선택하세요"] + inventory_update_menu, key='update_selectbox')
   
    inventory_management_menu = ["품번별 공정능력분석", "품번별 시계열 그래프"]
    inventory_choice = st.sidebar.selectbox('재고관리', ["Menu를 선택하세요"] + inventory_management_menu, key='inventory_selectbox')

    # 첫 페이지에 "협력사 재고 관리 시스템" 출력
    if 'page' not in st.session_state:
        st.session_state['page'] = "home"       

    if choice == "Menu를 선택하세요" and inventory_update_choice == "Menu를 선택하세요" and inventory_choice == "Menu를 선택하세요":
        st.session_state['page'] = "home"
    elif choice == "협력사별 재고현황":
        st.session_state['page'] = "partner"
    elif choice == "Part NO 별 재고현황":
        st.session_state['page'] = "part_no"
    elif inventory_update_choice == "재고 업데이트":
        st.session_state['page'] = "update_form"
    elif inventory_update_choice == "재고 추가":
        st.session_state['page'] = "add_form"
    elif inventory_update_choice == "재고 삭제":
        st.session_state['page'] = "delete_form"
    elif inventory_choice == "품번별 공정능력분석":
        st.session_state['page'] = "process_form"
    elif inventory_choice == "품번별 시계열 그래프":
        st.session_state['page'] = "time_series_graph"

    # 페이지에 따른 내용 표시    
    if st.session_state.get('page') == 'home':
        data = load_data()
        st.title('협력사 사내재고관리 시스템')
        search_query = st.text_input("검색어를 입력하세요")
        
        # 검색어가 입력된 경우 필터링 수행
        if search_query:
            filtered_data = data[data.apply(lambda row: search_query.lower() in row.astype(str).str.lower().values, axis=1)]
            if not filtered_data.empty:
                st.write(f"검색어 '{search_query}'에 해당하는 결과:")
                st.dataframe(filtered_data)
            else:
                st.write(f"검색어 '{search_query}'에 해당하는 결과가 없습니다.")
                         
    elif st.session_state.get('page') == 'partner':
        df = load_data()
        st.title("협력사별 재고현황")
        partner = st.selectbox("협력사를 선택하세요", df["company"].unique(), key='partner_selectbox')
        pfiltered_df = df[df['company'] == partner]
        st.write(pfiltered_df)
        if st.button("Home으로 돌아가기"):
            reset_session_state()
            st.experimental_rerun()
            
    elif st.session_state.get('page') == "part_no":
        df = load_data()
        st.title("PART NO별 재고현황")
        part_no = st.selectbox("PART Number를 선택하세요", df["PART_NO"].unique(), key='part_no_selectbox')
        filtered_df = df[df['PART_NO'] == part_no]
        st.write(filtered_df)
        if st.button("Home으로 돌아가기"):
            reset_session_state()
            st.experimental_rerun()
                
    elif st.session_state.get('page') == "update_form":
        st.title("재고 업데이트 폼")
        update_data()
        if st.button("Home으로 돌아가기"):
            reset_session_state()
            st.experimental_rerun()
            
    elif st.session_state.get('page') == "add_form":
        st.title("재고 추가 폼")
        add_data()
        if st.button("Home으로 돌아가기"):
            reset_session_state()
            st.experimental_rerun()
            
    elif st.session_state.get('page') == "delete_form":
        st.title("재고 삭제 폼")
        delete_data()
        if st.button("Home으로 돌아가기"):
            reset_session_state()
            st.experimental_rerun()
            
    elif st.session_state.get('page') == "process_form":
        st.title("공정능력 폼")
        calculate_Z_bench()
        if st.button("Home으로 돌아가기"):
            reset_session_state()
            st.experimental_rerun()
    elif st.session_state.get('page') == "time_series_graph":
        st.title("품번별 시계열 그래프 페이지")
        plot_time_series()
        if st.button("Home으로 돌아가지"):
            reset_session_state()
            st.experimental_rerun()


# In[28]:


if __name__ == "__main__":
    main()

