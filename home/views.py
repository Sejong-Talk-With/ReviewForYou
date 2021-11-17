from functions import *
from django.shortcuts import render, redirect
from .models import ProductModel, ReviewModel, ProductKeyword
import re
from viz_trend import *
from collections import Counter, defaultdict


def foward_home(request):
    if request.method == 'GET':
        T_Products = [ProductModel.objects.all().order_by('-search_value')[i * 5:i * 5 + 5] for i in range(4)]
        R_Products = [ProductModel.objects.all().order_by('-created_at')[i * 5:i * 5 + 5] for i in range(4)]
        return render(request, 'home/main.html', {'TOP_Products': T_Products, 'RECENT_Products': R_Products})


def home(request):
    if request.method == 'GET':
        T_Products = [ProductModel.objects.all().order_by('-search_value')[i * 5:i * 5 + 5] for i in range(4)]
        R_Products = [ProductModel.objects.all().order_by('-created_at')[i * 5:i * 5 + 5] for i in range(4)]
        first = 1
        return render(request, 'home/main.html',
                      {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'first': first})


def product_detail(request, id):
    T_Products = [ProductModel.objects.all().order_by('-search_value')[i * 5:i * 5 + 5] for i in range(4)]
    R_Products = [ProductModel.objects.all().order_by('-created_at')[i * 5:i * 5 + 5] for i in range(4)]
    click_Product = ProductModel.objects.get(pk=id)
    click_Product_keyword = ProductKeyword.objects.filter(product_id=id)
    all_comments = ReviewModel.objects.filter(product_id=id)  # .order_by('score')
    positive_comments = [c for c in all_comments if c.score >= 6.5 and len(c.review.split()) <= 40][:30]
    negative_comments = [c for c in all_comments if c.score <= 3.5 and len(c.review.split()) <= 40][:30]

    pos_review = []
    for pos in positive_comments:
        pos_result = [(w, v) for w, v in zip(pos.morph.split(), map(float, pos.xai_vale.split()))]
        pos_review.append([pos_result, pos])

    neg_review = []
    for neg in negative_comments:
        neg_result = [(w, v) for w, v in zip(neg.morph.split(), map(float, neg.xai_vale.split()))]
        neg_review.append([neg_result, neg])

    if request.method == 'GET':  # 'positive_comments':positive_comments, 'negative_comments':negative_comments,
        return render(request, 'home/modal.html',
                      {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'click_Product': click_Product,
                       'click_Product_keyword': click_Product_keyword, 'pos_review': pos_review,
                       'neg_review': neg_review})
    if request.method == 'POST':
        keywords = request.POST.getlist("rGroup", '')
        if keywords == '':
            return render(request, 'home/modal.html',
                          {'TOP_Products': T_Products, 'RECENT_Products': R_Products,'click_Product': click_Product,
                           'click_Product_keyword': click_Product_keyword, 'pos_review': pos_review,
                           'neg_review': neg_review})
        categories = ProductModel.objects.get(id=id).categories.split(', ')
        all_productmodel = ProductModel.objects.exclude(id=id)
        models = set()
        for model in all_productmodel:
            for cate in categories:
                if cate in model.categories:
                    models.add(model)
        result = {}
        # 상품 추천
        for model in models:
            total = 0
            cnt = 0
            model_keyword = ProductKeyword.objects.filter(product_id=model.id)
            if len(set(keywords) - set(keyword.keyword for keyword in model_keyword)) == 0:
                key_dict = {keyword.keyword: keyword.keyword_positive for keyword in model_keyword}
                line = []
                for word in keywords:
                    total += key_dict[word]
                    cnt += 1
                    line.append([word, key_dict[word]])
            if cnt != 0:
                average = total / cnt
                result[model] = [average, line]
        top3_products = [[m, v[1]] for m, v in sorted(result.items(), key=lambda x: x[1][0], reverse=True)[:3]]
        return render(request, 'home/modal.html',
                      {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'click_Product': click_Product,
                       'click_Product_keyword': click_Product_keyword, 'pos_review': pos_review,
                       'neg_review': neg_review, 'recom_model': top3_products})


def review_detail(request, review_id):
    T_Products = [ProductModel.objects.all().order_by('-search_value')[i * 5:i * 5 + 5] for i in range(4)]
    R_Products = [ProductModel.objects.all().order_by('-created_at')[i * 5:i * 5 + 5] for i in range(4)]
    click_review = ReviewModel.objects.get(pk=review_id)
    product_id = click_review.product_id.id
    click_Product = ProductModel.objects.get(pk=click_review.product_id.id)
    keywords = click_review.keywords.split()
    sentences = sss(click_review.review)

    keywords = list(map(lambda x: re.sub('[#]', '', x), keywords))

    if keywords:
        categories = ProductModel.objects.get(id=product_id).categories.split(', ')
        all_productmodel = ProductModel.objects.exclude(id=product_id)
        models = []
        for model in all_productmodel:
            if set(categories) & set(model.categories.split(', ')):
                models.append(model)
        result = {}
        for model in models:
            total, cnt = 0, 0
            model_keyword = ProductKeyword.objects.filter(product_id=model.id)
            if len(set(keywords) - set(keyword.keyword for keyword in model_keyword)) == 0:
                key_dict = {keyword.keyword: keyword.keyword_positive for keyword in model_keyword}
                line = []
                for word in keywords:
                    total += key_dict[word]
                    cnt += 1
                    line.append([word, key_dict[word]])
            if cnt != 0:
                average = total / cnt
                result[model] = [average, line]

        top3_products = [[m, v[1]] for m, v in sorted(result.items(), key=lambda x: x[1][0], reverse=True)[:3]]
    else:
        top3_products = []

    temp = []

    keyword_rate = defaultdict(float)
    keyword_cnt = defaultdict(int)
    for sen in sentences:
        ws, value = DNN_func(sen)
        line = [*map(lambda x, y: (x, y), ws, value)]
        # line=[]
        # value=[]
        # for w, v in zip(ws, vs_1):
        #     if len(w) == 0:
        #         break
        #     line.append((w, v))
        #     value.append(v)
        temp.append([line, sen])

        for word in keywords:
            similar_word = make_sim_word([word])
            if word in sen:  # 이 부분
                keyword_rate[word] += round(make_score(sum(value) / len(value)) * 100, 1)
                keyword_cnt[word] += 1
            else:
                if similar_word[word]:
                    for sim_word in similar_word[word]:
                        if sim_word in sen:
                            keyword_rate[word] += round(make_score(sum(value) / len(value)) * 100, 1)
                            keyword_cnt[word] += 1

    keyword_eval = []

    for w, total_rate in keyword_rate.items():
        keyword_eval.append([w, round((total_rate / keyword_cnt[w]), 2)])

    if request.method == 'GET':  # 'positive_comments':positive_comments, 'negative_comments':negative_comments,
        selected_review = sentences[0]
        other_reivew = ReviewModel.objects.filter(product_id=product_id).exclude(id=review_id)

        ws, value = DNN_func(selected_review)
        sen_xai = [*map(lambda x, y: (x, y), ws, value)]
        # sen_xai = []
        # value=[]
        # for w, v in zip(ws, vs_1):
        #     if len(w) == 0:
        #         break
        #     sen_xai.append((w, v))
        #     value.append(v)

        text = []
        for review in other_reivew:
            text.append(review.review)
        review_data, vocab_sorted = return_review_data(text)
        result_same_sentences, same_rate = result_of_selected_review_s_same_reviews(selected_review, round(
            make_score(sum(value) / len(value)) * 100, 2), review_data, vocab_sorted)

        return render(request, 'home/detail.html',
                      {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'click_Product': click_Product,
                       'click_review': click_review,
                       'recom_model': top3_products, 'result': temp, 'click_review_keywords': keyword_eval,
                       'similar_review': result_same_sentences, 'similar_rate': same_rate, 'sen_xai': sen_xai})

    if request.method == 'POST':
        selected_review = request.POST.get("review", '')
        other_reivew = ReviewModel.objects.filter(product_id=product_id).exclude(id=review_id)

        ws, value = DNN_func(selected_review)
        sen_xai = [*map(lambda x, y: (x, y), ws, value)]
        # sen_xai = []
        # value = []
        # for w, v in zip(ws, vs_1):
        #     if len(w) == 0:
        #         break
        #     sen_xai.append((w, v))
        #     value.append(v)

        text = []
        for review in other_reivew:
            text.append(review.review)
        review_data, vocab_sorted = return_review_data(text)
        result_same_sentences, same_rate = result_of_selected_review_s_same_reviews(selected_review, round(
            make_score(sum(value) / len(value)) * 100, 2), review_data, vocab_sorted)

        return render(request, 'home/detail.html',
                      {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'click_Product': click_Product,
                       'click_review': click_review,
                       'recom_model': top3_products, 'result': temp, 'click_review_keywords': keyword_eval,
                       'similar_review': result_same_sentences, 'similar_rate': same_rate, 'sen_xai': sen_xai})


def url_search(request):
    T_Products = [ProductModel.objects.all().order_by('-search_value')[i * 5:i * 5 + 5] for i in range(4)]
    R_Products = [ProductModel.objects.all().order_by('-created_at')[i * 5:i * 5 + 5] for i in range(4)]
    first = 1
    if request.method == 'POST':
        url_src = request.POST.get("url_src", '')
        if '/' in url_src:
            if 'share' in url_src:
                product_num = url_src.split('/')[-2].split('?')[0]
            else:
                product_num = url_src.split('/')[-1].split('?')[0]

            if url_src == '':
                return render(request, 'home/main.html',
                              {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'first': first})
            elif ProductModel.objects.filter(product_num=product_num).exists():
                search_product = ProductModel.objects.get(product_num=product_num)
                search_product.search_value += 1
                search_product.save()
                id = search_product.id
                return redirect('/modal/{}/'.format(id))
            else:
                if '11st' in url_src:
                    site = 1
                    tem_data, pre_product_name, img_src, price, categories, result, keyword, keyword_ratio = lets_do_crawling(
                        site, product_num)
                elif 'naver' in url_src:
                    site = 2
                    tem_data, pre_product_name, img_src, price, categories, result, keyword, keyword_ratio = lets_do_crawling(
                        site, product_num, url_src)
                else: # 지원하지 않는 URL 형식
                    return render(request, 'home/main.html',
                                  {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'first': first})

                product_name = []
                for sen in pre_product_name.split():
                    if '[' in sen or ']' in sen or '/' in sen:
                        pass
                    else:
                        product_name.append(sen)

                product_name = ' '.join(product_name).strip()

                xai_before_text = tem_data['xai_before_text']
                xai_value = tem_data['xai_value']
                xai_positive_negative = tem_data['xai_positive_negative']
                dates = tem_data['date']
                product = ProductModel.objects.create(product_url=url_src, product_name=product_name,
                                                      img_src=img_src, price=price,
                                                      categories=categories, product_num=product_num, search_value=1)
                product_score_list = []
                for index, row in tem_data.iterrows():
                    review_model = ReviewModel()
                    word_list = keyword_in_review(row['review'], keyword)
                    result_words = ''
                    if word_list:
                        result_words = '#' + " #".join(word_list).strip()
                    review_model.keywords = result_words
                    review_model.product_id = product
                    review_model.review = " ".join(xai_before_text[index])
                    tmp_score = xai_positive_negative[index]
                    review_model.score = tmp_score
                    product_score_list.append(tmp_score)
                    review_model.morph = " ".join(xai_before_text[index])
                    review_model.xai_vale = " ".join([*map(lambda x: str(x), xai_value[index])])
                    review_model.date = dates[index]
                    review_model.save()
                product.pos_neg_rate = int(sum([1 for i in product_score_list if i > 5]) * 100 / (index + 1))
                product.total_value = round(sum(product_score_list) / (index + 1), 1)
                product.save()

                for word, sentence in result.items():
                    product_keyword = ProductKeyword()
                    product_keyword.product_id = product
                    product_keyword.keyword = word
                    product_keyword.summarization = sentence
                    product_keyword.keyword_positive = round(float(keyword_ratio[word]), 1)
                    product_keyword.save()

            return render(request, 'home/main.html',
                          {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'first': first})

        else:
            search_cate = url_src
            if search_cate == '':
                return render(request, 'home/main.html',
                              {'error01': 'URL 혹은 카테고리를 입력해주세요!', 'TOP_Products': T_Products,
                               'RECENT_Products': R_Products})
            else:
                same_cate_products = []
                flag = 0
                for product in ProductModel.objects.all():
                    if search_cate in product.categories:
                        same_cate_products.append([product, product.search_value, product.created_at])
                        flag = 1
                if flag != 1:
                    return render(request, 'home/main.html',
                                  {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'first': first})
                # 카테고리 검색 시 오류, 수정필요
                T_Products = [
                    [p for p, _, _ in sorted(same_cate_products, key=lambda x: x[1], reverse=True)][i * 5:i * 5 + 5] for
                    i in range(4)]
                R_Products = [
                    [p for p, _, _ in sorted(same_cate_products, key=lambda x: x[2], reverse=True)][i * 5:i * 5 + 5] for
                    i in range(4)]
                return render(request, 'home/main.html',
                              {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'first': first})


def trend(request, id):
    All_Products = ProductModel.objects.all()

    click_categories = ProductModel.objects.get(pk=id).categories
    if "," in click_categories:
        click_categories = click_categories.split(", ")[0]
    arr = []
    date_arr = []
    for i in All_Products:
        if "," in i.categories:
            tmp = i.categories.split(",")[0]
        else:
            tmp = i.categories
        if click_categories == tmp:
            reviews = ReviewModel.objects.filter(product_id=i.id)
            for j in reviews:
                if len(j.keywords) == 0:
                    continue
                arr += j.keywords.split(" ")
                date_arr.append(j.date)
    make_charts(Counter(arr), date_arr)

    T_Products = [ProductModel.objects.all().order_by('-search_value')[i * 5:i * 5 + 5] for i in range(4)]
    R_Products = [ProductModel.objects.all().order_by('-created_at')[i * 5:i * 5 + 5] for i in range(4)]
    return render(request, 'home/trend.html',
                  {'TOP_Products': T_Products, 'RECENT_Products': R_Products, 'click_product': click_categories,
                   'id': id})


def delete_product(request, id):
    product = ProductModel.objects.get(id=id)
    product.delete()
    return redirect('/home')
