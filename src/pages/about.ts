import metrics from '../data/metrics.json';
import { h } from '../ui/dom';
import { lang, t } from '../i18n';

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

const copy = {
  ar: {
    how: 'كيف يعمل',
    steps: [
      'تحدد MediaPipe نقاط الجسم والوجه واليدين (يد واحدة أو اثنتين) في كل إطار من الكاميرا، على جهازك.',
      'يقرأ نموذج ST-Transformer حركة هذه النقاط عبر الزمن ويختار واحدة من 502 إشارة، ويقرأ نموذج أصغر شكل اليد في كل إطار ليكتب الحروف واليد ما زالت مرفوعة. في الوضع «التلقائي» يعملان معًا: اليد الساكنة المرفوعة تُقرأ حرفًا، والحركة تُقرأ كلمة، فيمكن خلط الكلمات والأسماء المهجّاة في جملة واحدة.',
      'تظهر الكلمة باهتة وأنت تشير، وتُثبَّت حين تنتقل إلى الإشارة التالية أو تُنزل يديك، ثم يمكن نطق الجملة بالصوت العربي في المتصفح.',
      'في الاتجاه المعاكس، يُطابق النص مع القاموس (مع تجاهل التشكيل واختلاف الهمزات والبادئات مثل «ال» و«و»)، وتُعرض إشارة مصوّرة لكل كلمة، وتُهجّى الكلمات غير المعروفة بالحروف.',
    ],
    accuracy: 'الدقة المقيسة',
    accuracyText: (m: typeof metrics) =>
      `اختُبر النموذج على ${m.videos} فيديو من قسم الاختبار في KArSL (${m.signers} مؤشرين، فيديو لكل إشارة). أصاب في المحاولة الأولى ${pct(m.top1)}، وكانت الإجابة الصحيحة ضمن أول خمسة اقتراحات ${pct(m.top5)}.`,
    sentencesText:
      'في جمل من ثلاث إشارات بمعدل 15 إطارًا في الثانية: يتعرف على نحو 93% من الكلمات إذا توقفت لحظة بين الإشارات، ونحو 54% إذا أشرت دون توقف؛ لذلك يساعد التوقف القصير كثيرًا. وفي جمل تخلط كلمتين وحرفين مهجّاين (الوضع التلقائي، مع توقف قصير): نحو 91%. لتهجئة الأسماء الطويلة بسرعة، وضع «حروف» أدق.',
    lettersText: (m: typeof metrics) =>
      `نموذج الحروف اختُبر على مؤشر لم يره أثناء التدريب (دُرّب على الاثنين الآخرين، وكُرر ذلك للثلاثة): أصاب ${pct(m.letters.videos)} من الحروف. أكثر ما يلتبس الحروف المتشابهة في الشكل ولا تختلف إلا بالحركة أو الهمزة، مثل ي/ى/ئ وت/ة وج/ح؛ اضغط على الحرف لاختيار البديل.`,
    caveat:
      'مؤشرو اختبار نموذج الكلمات ظهروا أيضًا في بيانات تدريبه، لذا ستكون دقته مع أشخاص جدد أقل. أفضل النتائج: إضاءة جيدة، الجزء العلوي من الجسم كاملًا داخل الصورة، وأداء الإشارة كما في القاموس.',
    language: 'اللغة',
    languageText:
      'يعتمد SLI لغة الإشارة العربية الموحّدة كما سُجّلت في قاعدة بيانات KArSL. قد تختلف بعض إشارات الكلمات عن اللهجات المحلية مثل لغة الإشارة المصرية.',
    credits: 'المصادر والشكر',
    team: 'فريق الأنامل الرقمية — إعداد وتطوير الطالبين: أنس محمد مختار، إياد عبد الرؤوف سمير.',
  },
  en: {
    how: 'How it works',
    steps: [
      'MediaPipe finds body, face and hand points (one hand or both) in each camera frame, on your device.',
      'An ST-Transformer reads how those points move over time and picks one of 502 signs, and a smaller model reads the hand shape in every frame to type letters while the hand is still up. In Auto mode both work together: a still, raised hand is read as a letter and movement as a word, so words and spelled names can be mixed in one sentence.',
      'A word shows faintly while you sign and is confirmed when you move on to the next sign or lower your hands; the sentence can then be spoken with the browser’s Arabic voice.',
      'The other way round, text is matched against the dictionary (ignoring diacritics, hamza spellings and prefixes like ال and و), each word is shown as a recorded sign, and unknown words are fingerspelled.',
    ],
    accuracy: 'Measured accuracy',
    accuracyText: (m: typeof metrics) =>
      `Tested on ${m.videos} videos from the KArSL test split (${m.signers} signers, one video per sign). The first guess was right ${pct(m.top1)} of the time, and the right answer was in the top five ${pct(m.top5)} of the time.`,
    sentencesText:
      'In three-sign sentences at 15 frames per second, about 93% of words are recognised with a brief pause between signs and about 54% when signing straight through, so a short pause helps a lot. Sentences mixing two words and two spelled letters (Auto mode, brief pauses): about 91%. For spelling long names quickly, Letters mode is more accurate.',
    lettersText: (m: typeof metrics) =>
      `The letter model was tested on a signer it never saw (trained on the other two, repeated for all three): it got ${pct(m.letters.videos)} of letters right. The usual mix-ups are letters with the same hand shape that differ only by movement or a hamza, such as ي/ى/ئ, ت/ة and ج/ح; tap a letter to pick the alternative.`,
    caveat:
      'The word model’s test signers also appear in its training data, so accuracy for new people will be lower. Best results: good light, your whole upper body in frame, and signs performed as in the dictionary.',
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
      h('p', {}, c.sentencesText),
      h('p', {}, c.lettersText(metrics)),
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
