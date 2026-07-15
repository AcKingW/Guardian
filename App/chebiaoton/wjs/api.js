/* ***************************incoming*********************************** */
function findById(params) { //面签页面--根据apid查询已有信息
    return service({
        url: '/api/backend/approvalProcess/findById',
        method: 'get',
        params
    })
}

function faceSignSave(params) { //面签页面--保存提交
    return service({
        url: '/api/backend/faceSign/save',
        method: 'post',
        data: JSON.stringify(params)
    })
}
