import metrics from '../data/metrics.json';
import { h } from '../ui/dom';
import { lang, t } from '../i18n';

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

const copy = {
  ar: {
    how: 'كيف يعمل',
    steps: [
      'تحدد MediaPipe نقاط الجسم والوجه واليد في كل إطار من الكاميرا، على جهازك.',
      'يقرأ نموذج ST-Transformer حركة هذه النقاط عبر الزمن ويختار واحدة من 502 إشارة: الأرقام والحروف والكلمات.',
      'تُثبَّت الكلمة عند إنزال اليدين، أو مبكرًا إذا ثبت النموذج على الإجابة نفسها بثقة عالية نحو ثانية، ثم تُنطق بالصوت العربي في المتصفح.',
      'في الاتجاه المعاكس، يُطابق النص مع القاموس (مع تجاهل التشكيل واختلاف الهمزات والبادئات مثل «ال» و«و»)، وتُعرض إشارة مصوّرة لكل كلمة، وتُهجّى الكلمات غير المعروفة بالحروف.',
    ],
    accuracy: 'الدقة المقيسة',
    accuracyText: (m: typeof metrics) =>
      `اختُبر النموذج على ${m.videos} فيديو من قسم الاختبار في KArSL (${m.signers} مؤشرين، فيديو لكل إشارة). أصاب في المحاولة الأولى ${pct(m.top1)}، وكانت الإجابة الصحيحة ضمن أول خمسة اقتراحات ${pct(m.top5)}.`,
    caveat:
      'هؤلاء المؤشرون أنفسهم ظهروا في بيانات التدريب، لذا ستكون الدقة مع أشخاص جدد أقل. أفضل النتائج: إضاءة جيدة، الجزء العلوي من الجسم كاملًا داخل الصورة، وأداء الإشارة كما في القاموس.',
    language: 'اللغة',
    languageText:
      'يعتمد SLI لغة الإشارة العربية الموحّدة كما سُجّلت في قاعدة بيانات KArSL. قد تختلف بعض إشارات الكلمات عن اللهجات المحلية مثل لغة الإشارة المصرية.',
    credits: 'المصادر والشكر',
    team: 'فريق الأنامل الرقمية — إعداد وتطوير الطالبين: أنس محمد مختار، إياد عبد الرؤوف سمير.',
  },
  en: {
    how: 'How it works',
    steps: [
      'MediaPipe finds body, face and hand points in each camera frame, on your device.',
      'An ST-Transformer model reads how those points move over time and picks one of 502 signs: numbers, letters and words.',
      'A word is committed when the hands drop, or earlier if the model stays confident in the same answer for about a second; it is then spoken with the browser’s Arabic voice.',
      'The other way round, text is matched against the dictionary (ignoring diacritics, hamza spellings and prefixes like ال and و), each word is shown as a recorded sign, and unknown words are fingerspelled.',
    ],
    accuracy: 'Measured accuracy',
    accuracyText: (m: typeof metrics) =>
      `Tested on ${m.videos} videos from the KArSL test split (${m.signers} signers, one video per sign). The first guess was right ${pct(m.top1)} of the time, and the right answer was in the top five ${pct(m.top5)} of the time.`,
    caveat:
      'The same signers appear in the training data, so accuracy for new people will be lower. Best results: good light, your whole upper body in frame, and signs performed as in the dictionary.',
    language: 'Language',
    languageText:
      'SLI uses Unified Arabic Sign Language as recorded in the KArSL database. Some word signs may differ from local variants such as Egyptian Sign Language.',
    credits: 'Sources and thanks',
    team: 'Digital Fingers team — prepared and developed by students Anas Mohamed Mokhtar and Iyad Abdel Raouf Samir.',
  },
};

export function mountAbout(root: HTMLElement) {
  const c = copy[lang()];
  root.replaceChildren(
    h(
      'div',
      { class: 'page page-about prose' },
      h('header', { class: 'page-head' }, h('h1', {}, t().aboutTitle), h('p', {}, t().tagline)),
      h('p', { class: 'callout' }, t().offline),
      h('h2', {}, c.how),
      h('ol', {}, ...c.steps.map((s) => h('li', {}, s))),
      h('h2', {}, c.accuracy),
      h('p', {}, c.accuracyText(metrics)),
      h('p', { class: 'hint' }, c.caveat),
      h('h2', {}, c.language),
      h('p', {}, c.languageText),
      h('h2', {}, c.credits),
      h('p', {}, c.team),
      h(
        'ul',
        { class: 'credits', dir: 'ltr' },
        h('li', {}, 'KArSL: Arabic Sign Language Database — Sidig, Luqman, Mahmoud, Mohandes. ACM TALLIP 20(1), 2021. Sign videos and labels.'),
        h('li', {}, 'Word-level ArSL ST-Transformer by Yousef Elkilany (MIT License). Pretrained recognition model.'),
        h('li', {}, 'MediaPipe by Google (Apache 2.0). Pose, face and hand tracking.'),
        h('li', {}, 'ONNX Runtime Web by Microsoft (MIT License).'),
      ),
    ),
  );
  return () => {};
}
