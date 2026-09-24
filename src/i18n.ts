export type UiLang = 'ar' | 'en';

const ar = {
  appName: 'SLI',
  tagline: 'مترجم لغة الإشارة العربية بالذكاء الاصطناعي',
  navSign: 'أشِر',
  navText: 'اكتب أو تكلّم',
  navTalk: 'محادثة',
  navDict: 'القاموس',
  navAbout: 'عن التطبيق',
  langToggle: 'English',
  themeToggle: 'تبديل المظهر',

  // sign -> text
  signTitle: 'أشِر أمام الكاميرا',
  signHint: 'اختر «كلمات» أو «حروف» ثم أشِر أمام الكاميرا؛ تظهر الترجمة وأنت تشير.',
  modeLabel: 'نوع الإشارة',
  modes: { words: 'كلمات', letters: 'حروف' } as Record<string, string>,
  modeHints: {
    words: 'أشِر بالكلمة؛ تظهر باهتة وأنت تشير ثم تُثبَّت حين يتأكد منها التطبيق.',
    letters: 'تهجَّ حرفًا حرفًا: اثبت على كل حرف لحظة ثم انتقل للتالي دون إنزال يدك. أنزل يدك لإنهاء الكلمة.',
  } as Record<string, string>,
  startCamera: 'تشغيل الكاميرا',
  stopCamera: 'إيقاف الكاميرا',
  loadingModels: 'جارٍ تحميل النماذج…',
  loadingStep: { pose: 'نموذج الجسم', face: 'نموذج الوجه', hand: 'نموذج اليد' } as Record<string, string>,
  cameraDenied: 'لم يُسمح باستخدام الكاميرا. اسمح بها من إعدادات المتصفح ثم أعد المحاولة.',
  cameraMissing: 'لم نجد كاميرا على هذا الجهاز.',
  cameraError: 'تعذر تشغيل الكاميرا',
  cameraOff: 'الكاميرا متوقفة',
  noHand: 'لا توجد يد في الصورة',
  handSeen: 'نرى يدك',
  maybe: 'ربما',
  speakSentence: 'انطق الجملة',
  undo: 'تراجع',
  clear: 'مسح',
  emptySentence: 'لم تُترجم أي إشارة بعد.',
  tapToFix: 'اضغط على كلمة لاختيار بديل.',
  alternatives: 'بدائل',
  remove: 'حذف',
  autoSpeak: 'انطق كل كلمة فور التعرف عليها',
  fps: 'إطار/ث',

  // text -> sign
  textTitle: 'اكتب أو تكلّم، وشاهد الإشارة',
  textPlaceholder: 'اكتب جملة… مثل: السلام عليكم، أنا طبيب',
  translate: 'ترجم إلى إشارة',
  listen: 'تكلّم',
  listening: 'نستمع…',
  stopListening: 'إيقاف الاستماع',
  sttUnsupported: 'التعرف على الكلام غير متاح في هذا المتصفح. جرّب Chrome أو Edge.',
  sttError: 'تعذر التعرف على الكلام',
  play: 'تشغيل',
  pause: 'إيقاف مؤقت',
  replay: 'إعادة',
  speed: 'السرعة',
  spelled: 'تهجئة بالحروف',
  noSign: 'لا توجد إشارة لهذا الرمز',
  planEmpty: 'اكتب كلمة أو جملة لترى إشاراتها.',
  examples: 'جرّب',

  // conversation
  talkTitle: 'محادثة بين أصم وسامع',
  talkDeaf: 'الأصم يشير',
  talkHearing: 'السامع يتكلم',
  talkSend: 'أرسل',
  talkEmpty: 'ابدأ المحادثة: أشِر أمام الكاميرا أو اضغط «تكلّم».',
  showInSign: 'اعرض بالإشارة',
  typeInstead: 'أو اكتب هنا…',

  // dictionary
  dictTitle: 'قاموس الإشارات',
  dictSearch: 'ابحث عن كلمة…',
  allCats: 'الكل',
  cats: {
    social: 'تحيات ومجاملات', family: 'العائلة والناس', verbs: 'أفعال', traits: 'صفات ومشاعر',
    places: 'اتجاهات وأماكن', home: 'البيت', health: 'الصحة والجسم', jobs: 'المهن',
    religion: 'الدين', numbers: 'الأرقام', letters: 'الحروف',
  } as Record<string, string>,
  signs: 'إشارة',
  noResults: 'لا توجد إشارة بهذا الاسم. جرّب كلمة أخرى أو تهجئتها بالحروف.',
  close: 'إغلاق',

  // about
  aboutTitle: 'عن SLI',
  offline: 'يعمل التعرف كله على جهازك؛ لا تُرسل صورة الكاميرا إلى أي خادم.',
};

type Dict = typeof ar;

const en: Dict = {
  appName: 'SLI',
  tagline: 'AI Arabic Sign Language interpreter',
  navSign: 'Sign',
  navText: 'Type or speak',
  navTalk: 'Conversation',
  navDict: 'Dictionary',
  navAbout: 'About',
  langToggle: 'العربية',
  themeToggle: 'Toggle theme',

  signTitle: 'Sign in front of the camera',
  signHint: 'Pick Words or Letters, then sign to the camera; the translation appears as you sign.',
  modeLabel: 'What are you signing?',
  modes: { words: 'Words', letters: 'Letters' },
  modeHints: {
    words: 'Sign a word: it appears faintly while you sign and is confirmed once the app is sure.',
    letters: 'Spell letter by letter: hold each letter a moment, then move to the next without lowering your hand. Lower your hand to end the word.',
  },
  startCamera: 'Start camera',
  stopCamera: 'Stop camera',
  loadingModels: 'Loading models…',
  loadingStep: { pose: 'body model', face: 'face model', hand: 'hand model' },
  cameraDenied: 'Camera access was blocked. Allow it in your browser settings and try again.',
  cameraMissing: 'No camera found on this device.',
  cameraError: 'Could not start the camera',
  cameraOff: 'Camera is off',
  noHand: 'No hand in view',
  handSeen: 'Hand in view',
  maybe: 'Maybe',
  speakSentence: 'Speak sentence',
  undo: 'Undo',
  clear: 'Clear',
  emptySentence: 'No signs translated yet.',
  tapToFix: 'Tap a word to pick an alternative.',
  alternatives: 'Alternatives',
  remove: 'Remove',
  autoSpeak: 'Speak each word as soon as it is recognised',
  fps: 'fps',

  textTitle: 'Type or speak, and watch it signed',
  textPlaceholder: 'Type a sentence… e.g. السلام عليكم، أنا طبيب',
  translate: 'Translate to sign',
  listen: 'Speak',
  listening: 'Listening…',
  stopListening: 'Stop listening',
  sttUnsupported: 'Speech recognition is not available in this browser. Try Chrome or Edge.',
  sttError: 'Speech recognition failed',
  play: 'Play',
  pause: 'Pause',
  replay: 'Replay',
  speed: 'Speed',
  spelled: 'Fingerspelled',
  noSign: 'No sign for this character',
  planEmpty: 'Type a word or sentence to see its signs.',
  examples: 'Try',

  talkTitle: 'Deaf and hearing conversation',
  talkDeaf: 'Deaf person signs',
  talkHearing: 'Hearing person speaks',
  talkSend: 'Send',
  talkEmpty: 'Start talking: sign in front of the camera or press Speak.',
  showInSign: 'Show in sign',
  typeInstead: 'or type here…',

  dictTitle: 'Sign dictionary',
  dictSearch: 'Search for a word…',
  allCats: 'All',
  cats: {
    social: 'Greetings', family: 'Family & people', verbs: 'Verbs', traits: 'Traits & feelings',
    places: 'Directions & places', home: 'Home', health: 'Health & body', jobs: 'Jobs',
    religion: 'Religion', numbers: 'Numbers', letters: 'Letters',
  },
  signs: 'signs',
  noResults: 'No sign with that name. Try another word, or fingerspell it.',
  close: 'Close',

  aboutTitle: 'About SLI',
  offline: 'All recognition runs on your device; camera images are never sent to a server.',
};

const dicts = { ar, en };

let current: UiLang = (() => {
  try {
    return (localStorage.getItem('sli-lang') as UiLang) || 'ar';
  } catch {
    return 'ar';
  }
})();

export const lang = () => current;
export const t = () => dicts[current];

export function setLang(l: UiLang) {
  current = l;
  try {
    localStorage.setItem('sli-lang', l);
  } catch {
    /* private mode */
  }
  document.documentElement.lang = l;
  document.documentElement.dir = l === 'ar' ? 'rtl' : 'ltr';
}
