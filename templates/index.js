$("form").hide();

$(".testClick").click(function (){
  $("form").show(500);
  $(".testClick").hide(500);
});
