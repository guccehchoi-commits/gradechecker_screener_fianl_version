import streamlit as st

_PERF = {
    0.20: ('~85%', '~55%'), 0.25: ('~82%', '~60%'), 0.30: ('~80%', '~65%'),
    0.35: ('~77%', '~70%'), 0.40: ('72.9%', '73.8%'), 0.45: ('~70%', '~77%'),
    0.50: ('66.7%', '83.9%'), 0.55: ('~62%', '~87%'), 0.60: ('~57%', '~90%'),
    0.65: ('~52%', '~92%'), 0.70: ('~47%', '~94%'), 0.75: ('~44%', '~95%'),
    0.80: ('41.7%', '95.7%'),
}


@st.dialog('탐지 민감도 변경 확인')
def _confirm_threshold_change(new_thr: float):
    st.write(f'기준값(Threshold)을 **{new_thr:.2f}** 로 반영하시겠습니까?')
    st.caption('반영 시 예측 결과 · 예측 결과 분석 등 모든 화면에 즉시 적용됩니다.')
    c1, c2 = st.columns(2)
    with c1:
        if st.button('예, 반영합니다', type='primary', use_container_width=True):
            st.session_state['global_thr'] = new_thr
            st.session_state['thr_slider_input'] = new_thr
            st.session_state['_thr_pending'] = None
            st.rerun()
    with c2:
        if st.button('아니오', use_container_width=True):
            st.session_state['thr_slider_input'] = st.session_state.get('global_thr', 0.40)
            st.session_state['_thr_pending'] = None
            st.rerun()


def render_threshold_sidebar() -> float:
    """모든 페이지 사이드바에 공통으로 렌더링되는 탐지 민감도(Threshold) 설정 위젯.
    반환값은 '확인' 절차를 거쳐 실제로 반영된 기준값이다."""
    st.sidebar.subheader('⚙️ 탐지 민감도 설정')
    st.sidebar.caption('값을 낮추면 더 많은 게임을 검토 대상으로 잡아냅니다.')

    confirmed_thr = st.session_state.get('global_thr', 0.40)

    picked = st.sidebar.slider(
        '기준값 (Threshold)',
        min_value=0.20, max_value=0.80, value=confirmed_thr, step=0.05,
        key='thr_slider_input',
    )

    if picked != confirmed_thr:
        st.session_state['_thr_pending'] = picked

    if st.session_state.get('_thr_pending') is not None:
        st.sidebar.warning(f"⏳ 변경 대기 중: **{st.session_state['_thr_pending']:.2f}** (아직 미반영)")
        if st.sidebar.button('✅ 변경 사항 적용', use_container_width=True):
            _confirm_threshold_change(st.session_state['_thr_pending'])

    thr = confirmed_thr  # 확인 전까지는 기존에 반영된 값을 그대로 사용

    if thr <= 0.30:
        st.sidebar.info('**폭넓게 탐지**\n\n의심스러운 게임을 최대한 잡아냅니다. 검토할 게임 수가 다소 많아질 수 있습니다.')
    elif thr <= 0.45:
        st.sidebar.success('**권장 설정 ★**\n\n탐지 범위와 정확도의 균형입니다.\n실제 재분류 게임의 약 **70%** 를 잡아냅니다.')
    elif thr <= 0.65:
        st.sidebar.warning('**정밀 탐지**\n\n확실한 게임만 포함합니다. 경계선 게임을 놓칠 수 있습니다.')
    else:
        st.sidebar.error('**고정밀 모드**\n\n매우 확실한 게임만 탐지합니다. 탐지율이 크게 낮아집니다.')

    rec, pre = _PERF.get(round(thr, 2), ('—', '—'))
    st.sidebar.caption(f'예상 재현율 {rec} / 정밀도 {pre}')
    st.sidebar.caption(f'✅ 현재 반영된 기준값: **{thr:.2f}**')
    st.sidebar.divider()
    st.sidebar.caption('※ 예측 결과는 우선 재검토 순위 제공 목적')
    st.sidebar.caption('※ 최종 판정은 전문가 검토 필수')

    return thr
