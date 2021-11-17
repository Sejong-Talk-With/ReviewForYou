
function show_modal(){
    var modal = document.querySelector('.modal');
    modal.style.display ='block';

    var modal_out = document.querySelector('.modal_out');
    modal_out.style.display ='block';

}

function remove_modal(){
    var modal = document.querySelector('.modal');
    modal.style.display ='none';

    var modal_out = document.querySelector('.modal_out');
    modal_out.style.display ='none';

}

/*디테일*/
function show_detail(){
    var detail = document.querySelector('.detail');
    detail.style.display='block';

    var detail_out = document.querySelector('.detail_out');
    detail_out.style.display='block';

    var modal = document.querySelector('.modal');
    modal.style.display ='none';

    var modal_out = document.querySelector('.modal_out');
    modal_out.style.display ='none';
}

function remove_detail(){
    var detail = document.querySelector('.detail');
    detail.style.display='none';

    var detail_out = document.querySelector('.detail_out');
    detail_out.style.display='none';

    show_modal();
}

/*트렌드*/
function show_trend(){
    var trend = document.querySelector('.trend');
    trend.style.display='block';

    var trend_out = document.querySelector('.trend_out');
    trend_out.style.display='block';

    var modal = document.querySelector('.modal');
    modal.style.display='none';

    var modal_out = document.querySelector('.modal_out');
    modal_out.style.display='none';
}

function remove_trend(){
    var trend = document.querySelector('.trend');
    trend.style.display='none';

    var trend_out = document.querySelector('.trend_out');
    trend_out.style.display='none';


    var modal = document.querySelector('.modal');
    modal.style.display='block';

    var modal_out = document.querySelector('.modal_out');
    modal_out.style.display='block';

}


 function color1(i, classname,colorname){
    $(classname).css({
         "background":"conic-gradient("+colorname+" 0% "+i+"%, #1C1C1C "+i+"% 100%)"
    });
 }

 function get_data(num){
     console.log(num);
 }